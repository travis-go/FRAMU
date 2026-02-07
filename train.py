import os
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
device=torch.device("cuda:{}".format(0))

from torch.utils.data import DataLoader
import timm
from dataset import NPY_datasets
from tensorboardX import SummaryWriter
# Import FractalMultiScale model
from model import FractalMultiScaleUNetABL
# Import knowledge base (if needed)
from knowledge_base import AdaptiveKnowledgeBase

from engine import *

import sys

from utils import *
from config import setting_config

import warnings
warnings.filterwarnings("ignore")


# Define a FractalMultiScale wrapper to make it compatible with LBUNet interface
class FractalMultiScaleWrapper(torch.nn.Module):
    def __init__(self, model, use_fusion=True):
        super(FractalMultiScaleWrapper, self).__init__()
        self.model = model
        self.use_fusion = use_fusion  # Whether to use fusion mode
    
    def forward(self, x):
        # Always get two outputs for training
        if self.training:
            # Training: get intuitive and reflection outputs for loss calculation
            intuitive_output, reflection = self.model(x, use_fusion=False)
            output = intuitive_output  # Use intuitive output for main loss calculation during training
        else:
            # Inference: use fusion mode
            if self.use_fusion:
                fused_output, reflection, intuitive_output, final_output = self.model(x, use_fusion=True)
                output = fused_output
            else:
                output, reflection = self.model(x, use_fusion=False)
                intuitive_output = output
        
        # Ensure output is in [0,1] range
        output = torch.sigmoid(output)  # Add sigmoid to ensure output is in [0,1] range
        reflection = torch.sigmoid(reflection) if reflection is not None else reflection
        
        # Check output size to ensure it matches expectation
        # Assume target size should be [B, 1, H, W]
        if output.shape[1] > 1:
            # If output is multi-class, only take the first channel (assumed to be foreground class)
            output = output[:, 0:1]
        
        # Create a 5-element tuple as gt_pre
        # Ensure each element has correct size
        gt_pre = (output, output, output, output, output)
        
        # Create a 12-element tuple as key_points
        # First element is reflection vector, rest are duplicates
        key_points = (reflection, reflection, reflection, reflection, reflection, reflection,
                     reflection, reflection, reflection, reflection, reflection, reflection)
        
        # Final output
        out = output
        
        return gt_pre, key_points, out


# Define ABL loss function for FractalMultiScale
class FractalMultiScaleABLLoss(nn.Module):
    def __init__(self, wb=1, wd=1, reflection_loss_weight=1.0, consistency_loss_weight=1.0, 
                 size_loss_ratio=0.8, aux_seg_weight=0.0):
        super(FractalMultiScaleABLLoss, self).__init__()
        self.bcedice = BceDiceLoss(wb, wd)  # Base segmentation loss
        self.reflection_loss_weight = reflection_loss_weight
        self.consistency_loss_weight = consistency_loss_weight
        self.size_loss_ratio = size_loss_ratio  # Control the size of reflection regions
        self.aux_seg_weight = aux_seg_weight  # Auxiliary segmentation loss weight
        self.last_losses = None  # Store recent loss components
    
    def forward(self, gt_pre, key_points, out, target, points=None, knowledge_base=None, epoch=0, total_epochs=300):
        """
        Simplified ABL loss function focusing on reflection vector accuracy
        
        Args:
            gt_pre: Multi-scale predictions from model
            key_points: Key point/reflection vector predictions from model
            out: Final segmentation output
            target: Ground truth segmentation mask
            points: Boundary point annotations (may be None)
            knowledge_base: Knowledge base object
            epoch: Current training epoch
            total_epochs: Total training epochs
        """
        # 1. Calculate base segmentation loss
        main_loss = self.bcedice(out, target)
        
        # If no knowledge base or reflection vector, only return base loss
        if knowledge_base is None or key_points is None or len(key_points) == 0:
            self.last_losses = (main_loss.item(), 0.0, 0.0, 0.0)
            return main_loss
        
        # 2. Get reflection vector (first key point)
        reflection = key_points[0]
        
        # 3. Progressive reflection factor (gradually increase reflection weight during training)
        progression_factor = min(1.0, 0.1 + (epoch / (total_epochs // 2)))
        
        # 4. Reflection vector should indicate hard example regions
        # Calculate prediction error as supervision signal
        prediction_error = torch.abs(out - target)  # [B, 1, H, W]
        
        # 5. Reflection loss: reflection vector should output low confidence where error is large
# reflection(1 - error)，
        target_reflection = 1.0 - prediction_error
        reflection_loss = F.mse_loss(reflection, target_reflection)
        
        # 6. Size constraint loss: avoid reflection vector outputting all low confidence
        # Encourage reflection mean to approach a reasonable value (e.g. 0.7)
        mean_reflection = torch.mean(reflection)
        size_loss = F.mse_loss(mean_reflection, torch.tensor(0.7, device=reflection.device))
        
        # 7. Auxiliary segmentation loss (optional): additional supervision on high confidence regions
        aux_loss = 0.0
        if self.aux_seg_weight > 0:
            # High confidence regions should predict accurately
            confidence_mask = (reflection >= 0.6).float()
            weighted_error = prediction_error * confidence_mask
            aux_loss = torch.mean(weighted_error)
        
        # 8. Combine all losses
        weighted_reflection_loss = self.reflection_loss_weight * reflection_loss * progression_factor
        weighted_consistency_loss = self.consistency_loss_weight * size_loss * progression_factor
        weighted_aux_loss = self.aux_seg_weight * aux_loss if self.aux_seg_weight > 0 else 0.0
        
        total_loss = main_loss + weighted_reflection_loss + weighted_consistency_loss + weighted_aux_loss
        
        # Store loss components for logging
        main_loss_item = main_loss.item()
        reflection_loss_item = weighted_reflection_loss.item() if isinstance(weighted_reflection_loss, torch.Tensor) else weighted_reflection_loss
        consistency_loss_item = weighted_consistency_loss.item() if isinstance(weighted_consistency_loss, torch.Tensor) else weighted_consistency_loss
        aux_loss_item = weighted_aux_loss.item() if isinstance(weighted_aux_loss, torch.Tensor) else weighted_aux_loss
        
        self.last_losses = (main_loss_item, reflection_loss_item, consistency_loss_item, aux_loss_item)
        
        return total_loss


def main(config):

    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)



    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    set_seed(config.seed)
    torch.cuda.empty_cache()



    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = DataLoader(train_dataset,
                              batch_size=config.batch_size, 
                              shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            pin_memory=True, 
                            num_workers=config.num_workers,
                            drop_last=False)



    print('#----------Preparing Model----------#')
    model_cfg = config.model_config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if config.network == 'fractalmultiscale':
# FractalMultiScale
        fractalmultiscale_model = FractalMultiScaleUNetABL(
            in_channels=model_cfg.get('input_channels', 3),
            out_channels=model_cfg.get('num_classes', 1),
            base_channels=model_cfg.get('base_channels', 6),
            fractal_depth=model_cfg.get('fractal_depth', 3),
            patch_size=model_cfg.get('patch_size', 2),
            expand_ratio=model_cfg.get('expand_ratio', 1.5),
            d_state=model_cfg.get('d_state', 16),
            reflection_threshold=model_cfg.get('reflection_threshold', 0.5)
        )
        
# （ABL）
        knowledge_base = AdaptiveKnowledgeBase(
            device=device,
            default_class_names=['background', 'lesion'],
            shape_constraints={
                'continuity': model_cfg.get('continuity', 0.8),
                'compactness': model_cfg.get('compactness', 0.7),
                'smoothness': model_cfg.get('smoothness', 0.7)
            }
        )
        fractalmultiscale_model.set_knowledge_base(knowledge_base)
        print("Knowledge Base initialized for ABL")
        
# FractalMultiScaleLBUNet
        use_fusion = model_cfg.get('use_fusion', True) if 'use_fusion' in model_cfg else (config.abl.get('use_fusion', True) if hasattr(config, 'abl') else True)
        model = FractalMultiScaleWrapper(fractalmultiscale_model, use_fusion=use_fusion)
        print(f"Wrapper initialized with fusion mode: {use_fusion}")
        
# ABL
        if hasattr(config, 'abl'):
            abl_config = config.abl
            config.criterion = FractalMultiScaleABLLoss(
                wb=1, wd=1,  # BceDice
                reflection_loss_weight=abl_config.get('reflection_loss_weight', 1.0),
                consistency_loss_weight=abl_config.get('consistency_loss_weight', 1.0),
                size_loss_ratio=abl_config.get('size_loss_ratio', 0.8),
                aux_seg_weight=abl_config.get('auxiliary_segmentation_loss', 0.0)
            )
            print("Using ABL loss function with reflection mechanism")
    else:
        raise Exception('network is not right!')
    
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")


    print('#----------Preparing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)


    print('#----------Set other params----------#')
    min_value = 999
    start_epoch = 1
    min_epoch = 1


    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):

        torch.cuda.empty_cache()

# train_one_epoch
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# engine.pytrain_one_epoch，ABL
        step = train_one_epoch(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch,
            step,
            logger,
            config,
            writer,
            knowledge_base=knowledge_base,
            total_epochs=config.epochs      # （）
        )

# engine.pyval_one_epoch
        value = val_one_epoch(
            val_loader,
            model,
            config.criterion,
            epoch,
            logger,
            config,
            knowledge_base=knowledge_base
        )

        if value < min_value:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_value = value
            min_epoch = epoch


    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        test_one_epoch(
            val_loader,
            model,
            config.criterion,
            logger,
            config,
            path='ultimate',
            knowledge_base=knowledge_base
        )
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}.pth')
        )    


if __name__ == '__main__':
    config = setting_config
    main(config)
