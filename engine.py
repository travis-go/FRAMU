import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast
from sklearn.metrics import confusion_matrix
from utils import save_imgs


def train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer, knowledge_base=None, total_epochs=300):
    model.train()
    
    # Get device where model is located
    device = next(model.parameters()).device
    print(f"Training on device: {device}")
    
    # Track various losses
    loss_list = []
    reflection_loss_list = []  # ABL reflection loss
    consistency_loss_list = []  # ABL consistency loss
    aux_loss_list = []  # ABL auxiliary loss

    for iter, data in enumerate(train_loader):
        step += iter
        optimizer.zero_grad()
        images, targets, points = data
        if iter == 0:  # Only print information for the first batch
            print(f"Input data device before moving: img={images.device}, mask={targets.device}")
        images = images.to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        if points is not None:
            points = points.to(device, dtype=torch.float32)

        # Forward pass
        gt_pre, key_points, out = model(images)
        
        # Calculate loss - pass additional parameters needed for ABL
        if knowledge_base is not None and hasattr(criterion, 'reflection_loss_weight'):
            # If ABL loss function with knowledge base
            loss = criterion(gt_pre, key_points, out, targets, points, 
                             knowledge_base=knowledge_base, epoch=epoch, total_epochs=total_epochs)
        else:
            # Regular loss function
            loss = criterion(gt_pre, key_points, out, targets, points)

        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update parameters
        optimizer.step()
        
        loss_list.append(loss.item())

        # Current learning rate
        now_lr = optimizer.param_groups[0]['lr']

        # Log loss to tensorboard
        writer.add_scalar('loss/total', loss, global_step=step)
        
        # If using ABL, log component losses
        if knowledge_base is not None and hasattr(criterion, 'reflection_loss_weight'):
            # Assume ABL loss function can return component losses (main loss, reflection loss, etc.)
            if hasattr(criterion, 'last_losses') and criterion.last_losses is not None:
                main_loss, reflection_loss, consistency_loss, aux_loss = criterion.last_losses
                reflection_loss_list.append(reflection_loss)
                consistency_loss_list.append(consistency_loss)
                aux_loss_list.append(aux_loss)
                
                # Log to tensorboard
                writer.add_scalar('loss/reflection', reflection_loss, global_step=step)
                writer.add_scalar('loss/consistency', consistency_loss, global_step=step)
                writer.add_scalar('loss/auxiliary', aux_loss, global_step=step)

        # Print training information
        if iter % config.print_interval == 0:
            log_info = f'train: epoch {epoch}, iter:{iter}, loss: {np.mean(loss_list):.4f}, lr: {now_lr}'
            
            # If ABL loss components exist, add to log
            if reflection_loss_list:
                log_info += f', refl_loss: {np.mean(reflection_loss_list):.4f}'
            if consistency_loss_list:
                log_info += f', consist_loss: {np.mean(consistency_loss_list):.4f}'
            if aux_loss_list:
                log_info += f', aux_loss: {np.mean(aux_loss_list):.4f}'
                
            print(log_info)
            logger.info(log_info)
            
    # Update learning rate and log
    try:
        # Print learning rate before update
        old_lr = optimizer.param_groups[0]['lr']
        
        # Use different update strategies for different scheduler types
        if hasattr(scheduler, 'step') and callable(scheduler.step):
            # ReduceLROnPlateau needs validation metric
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                mean_loss = np.mean(loss_list)
                scheduler.step(mean_loss)
                print(f"ReduceLROnPlateau scheduler updated, validation loss: {mean_loss:.4f}")
            else:
                # Other schedulers call step() directly
                scheduler.step()
                print(f"Scheduler {type(scheduler).__name__} updated")
        
        # Print learning rate after update
        new_lr = optimizer.param_groups[0]['lr']
        if old_lr != new_lr:
            print(f"Learning rate updated: {old_lr:.6f} -> {new_lr:.6f}")
        else:
            print(f"Learning rate unchanged: {old_lr:.6f}")
            
    except Exception as e:
        print(f"Warning: Learning rate scheduler update failed: {e}")
        
    return step


def val_one_epoch(test_loader, model, criterion, epoch, logger, config, knowledge_base=None):
    model.eval()
    loss_list = []
    total_miou = 0.0
    total = 0
    gt_list = []
    pred_list = []
    
    with torch.no_grad():
        for data in tqdm(test_loader):
            # Compatible with both (img, msk) and (img, msk, points) return formats
            if len(data) == 3:
                img, msk, points = data
            else:
                img, msk = data
                points = None

            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()
            if points is not None:
                points = points.cuda(non_blocking=True).float()

            # Forward pass (wrapper already applied fusion mechanism during inference)
            gt_pre, key_points, out = model(img)

            # Calculate validation loss
            if knowledge_base is not None and hasattr(criterion, 'reflection_loss_weight'):
                # ABL loss: pass knowledge base consistent with training
                loss = criterion(
                    gt_pre,
                    key_points,
                    out,
                    msk,
                    points,
                    knowledge_base=knowledge_base,
                    epoch=epoch,
                    total_epochs=getattr(config, 'epochs', 300)
                )
            else:
                # Regular loss (e.g. GT_BceDiceLoss)
                loss = criterion(gt_pre, key_points, out, msk, points)

            loss_list.append(loss.item())
            
            # If knowledge base exists and in ABL mode, optionally apply additional knowledge base correction
            if knowledge_base is not None and hasattr(model, 'model') and hasattr(model.model, 'apply_abduction'):
                # Get reflection vector
                reflection = key_points[0]
                reflection_threshold = config.abl.get('reflection_threshold', 0.5) if hasattr(config, 'abl') else 0.5
                
                # Create mask - positions below threshold are marked for correction
                mask = (reflection < reflection_threshold).float()
                
                # Note: wrapper already performed fusion, can optionally apply knowledge base correction here
                if hasattr(config, 'abl') and config.abl.get('apply_kb_correction', False):
                    out_corrected = model.model.apply_abduction(out, mask, knowledge_base.perform_abduction)
                    out = out_corrected

            # Calculate metrics
            gts = msk.squeeze(1).cpu().detach().numpy()
            preds = out.squeeze(1).cpu().detach().numpy()
            
            gt_list.append(gts)
            pred_list.append(preds)
            
            preds_flat = np.array(preds).reshape(-1)
            gts_flat = np.array(gts).reshape(-1)

            y_pre = np.where(preds_flat >= config.threshold, 1, 0)
            y_true = np.where(gts_flat >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)
            
            total_miou += miou
            total += 1
            
    # Calculate overall metrics
    total_miou = total_miou / total
    pred_list = np.array(pred_list).reshape(-1)
    gt_list = np.array(gt_list).reshape(-1)

    y_pre = np.where(pred_list >= 0.5, 1, 0)
    y_true = np.where(gt_list >= 0.5, 1, 0)
    confusion = confusion_matrix(y_true, y_pre)
    TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

    accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
    sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
    specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
    f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0

    # Print validation results
    mean_loss = float(np.mean(loss_list)) if len(loss_list) > 0 else float('nan')
    log_info = f'val epoch: {epoch}, loss: {mean_loss:.4f}, miou: {total_miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}'
    
    # If using ABL correction, log it
    if knowledge_base is not None:
        log_info += f', with ABL correction (thresh: {config.abl.get("reflection_threshold", 0.5):.2f})'
        
    print(log_info)
    logger.info(log_info)
    
    # Return negative value for optimizer to minimize
    return - (total_miou + f1_or_dsc)


def test_one_epoch(test_loader, model, criterion, logger, config, path, test_data_name=None, knowledge_base=None):
    model.eval()
    gt_list = []
    pred_list = []
    total_miou = 0.0
    total = 0
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_loader)):
            img, msk, _ = data
            img, msk = img.cuda(non_blocking=True).float(), msk.cuda(non_blocking=True).float()

            # Forward pass (wrapper already applied fusion mechanism during inference)
            gt_pre, key_points, out = model(img)
            
            # If knowledge base exists and in ABL mode, optionally apply additional knowledge base correction
            if knowledge_base is not None and hasattr(model, 'model') and hasattr(model.model, 'apply_abduction'):
                # Get reflection vector
                reflection = key_points[0]
                reflection_threshold = config.abl.get('reflection_threshold', 0.5) if hasattr(config, 'abl') else 0.5
                
                # Create mask - positions below threshold are marked for correction
                mask = (reflection < reflection_threshold).float()
                
                # Note: wrapper already performed fusion, can optionally apply knowledge base correction here
                if hasattr(config, 'abl') and config.abl.get('apply_kb_correction', False):
                    out_corrected = model.model.apply_abduction(out, mask, knowledge_base.perform_abduction)
                    out = out_corrected

            # Convert to NumPy arrays
            msk = msk.squeeze(1).cpu().detach().numpy()
            out = out.squeeze(1).cpu().detach().numpy()

            gt_list.append(msk)
            pred_list.append(out)
            
            y_pre = np.where(out >= config.threshold, 1, 0)
            y_true = np.where(msk >= 0.5, 1, 0)

            smooth = 1e-5
            intersection = (y_pre & y_true).sum()
            union = (y_pre | y_true).sum()
            miou = (intersection + smooth) / (union + smooth)

            total_miou += miou
            total += 1

            # Save prediction visualization every save_interval samples
            if i % config.save_interval == 0:
                # If using ABL, also save reflection vector
                if knowledge_base is not None and hasattr(model, 'model'):
                    if test_data_name is None:
                        test_data_name = 'ISIC2017'
                    save_path = config.work_dir + 'outputs/' + test_data_name + '/'
                    
                    # Save original image, ground truth mask, prediction result, and reflection vector
                    reflection_map = key_points[0].cpu().detach().numpy()
                    
                    # Should call save_imgs function from utils.py, ensure it supports saving reflection vector
                    # If save_imgs doesn't support it, may need to modify it or add new save function
                    save_imgs(img, msk, out, key_points, gt_pre, i, save_path, config.datasets, 
                              config.threshold)
                else:
                    # Not using ABL, only save regular results
                    if test_data_name is None:
                        test_data_name = 'ISIC2017'
                    save_path = config.work_dir + 'outputs/' + test_data_name + '/'
                    save_imgs(img, msk, out, key_points, gt_pre, i, save_path, config.datasets, config.threshold)

        # Calculate overall metrics
        total_miou = total_miou / total

        pred_list = np.array(pred_list).reshape(-1)
        gt_list = np.array(gt_list).reshape(-1)

        y_pre = np.where(pred_list >= 0.5, 1, 0)
        y_true = np.where(gt_list >= 0.5, 1, 0)
        confusion = confusion_matrix(y_true, y_pre)
        TN, FP, FN, TP = confusion[0,0], confusion[0,1], confusion[1,0], confusion[1,1] 

        accuracy = float(TN + TP) / float(np.sum(confusion)) if float(np.sum(confusion)) != 0 else 0
        sensitivity = float(TP) / float(TP + FN) if float(TP + FN) != 0 else 0
        specificity = float(TN) / float(TN + FP) if float(TN + FP) != 0 else 0
        f1_or_dsc = float(2 * TP) / float(2 * TP + FP + FN) if float(2 * TP + FP + FN) != 0 else 0
        
        # Print test results
        log_info = f'test of best model, miou: {total_miou:.4f}, f1_or_dsc: {f1_or_dsc:.4f}'
        
        # If using ABL correction, add to log
        if knowledge_base is not None:
            log_info += f', with ABL correction (thresh: {config.abl.get("reflection_threshold", 0.5):.2f})'
            
        # Add other metrics
        log_info += f', accuracy: {accuracy:.4f}, sensitivity: {sensitivity:.4f}, specificity: {specificity:.4f}'
        
        print(log_info)
        logger.info(log_info)
        
        # Save test results to file
        result_file = config.work_dir + 'test_results.txt'
        with open(result_file, 'w') as f:
            f.write(f"Test Results:\n")
            f.write(f"Model: FractalMultiScale with {'ABL' if knowledge_base is not None else 'Standard'} training\n")
            f.write(f"Dataset: {config.datasets}\n")
            f.write(f"mIoU: {total_miou:.4f}\n")
            f.write(f"F1/DSC: {f1_or_dsc:.4f}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Sensitivity/Recall: {sensitivity:.4f}\n")
            f.write(f"Specificity: {specificity:.4f}\n")
            if knowledge_base is not None:
                f.write(f"Used ABL correction with reflection threshold: {config.abl.get('reflection_threshold', 0.5):.2f}\n")
        
        return {
            'miou': total_miou,
            'f1_or_dsc': f1_or_dsc,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity
        }
