import torch
import torch.nn.functional as F
import numpy as np

class AdaptiveKnowledgeBase:
    """
    Adaptive Knowledge Base: Provides domain knowledge-based constraints and reasoning mechanisms
    for self-reflection and abductive reasoning correction in medical image segmentation
    """
    
    def __init__(self, device, default_class_names=None, shape_constraints=None):
        """
        Initialize adaptive knowledge base
        
        Args:
            device: Computing device (CPU/GPU)
            default_class_names: Default class name list, e.g., ['background', 'lesion']
            shape_constraints: Shape constraint parameter dictionary, e.g.:
                {
                    'continuity': 0.2,  # Continuity constraint strength (0-1)
                    'compactness': 0.8,  # Compactness constraint strength (0-1)
                    'smoothness': 0.8,   # Smoothness constraint strength (0-1)
                }
        """
        self.device = device
        self.default_class_names = default_class_names if default_class_names else ['background', 'foreground']
        self.shape_constraints = shape_constraints if shape_constraints else {
            'continuity': 0.2,
            'compactness': 0.8,
            'smoothness': 0.8
        }
        
        # Initialize constraint strengths
        self.continuity_weight = self.shape_constraints.get('continuity', 0.2)
        self.compactness_weight = self.shape_constraints.get('compactness', 0.8)
        self.smoothness_weight = self.shape_constraints.get('smoothness', 0.8)
        
        # Initialize convolution kernels for morphological operations
        self.init_kernels()
        
    def init_kernels(self):
        """Initialize convolution kernels for morphological operations"""
        # Dilation and erosion kernels
        self.dilate_kernel = torch.ones(5, 5).to(self.device)
        self.erode_kernel = torch.ones(3, 3).to(self.device)
        
        # Edge detection kernel
        self.edge_kernel = torch.tensor([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=torch.float32).to(self.device)
        
        # Smoothing kernel
        self.smooth_kernel = torch.tensor([
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1]
        ], dtype=torch.float32).to(self.device) / 16.0

    def check_consistency(self, predictions, class_names=None):
        """
        Check consistency of predictions with domain knowledge
        
        Args:
            predictions: Model predictions [B, C, H, W]
            class_names: Class name list
            
        Returns:
            Consistency scores [B] - consistency score for each sample
        """
        if class_names is None:
            class_names = self.default_class_names
            
        batch_size = predictions.shape[0]
        consistency_scores = torch.zeros(batch_size).to(self.device)
        
        # Get hard segmentation results
        if predictions.shape[1] > 1:  # Multi-class case
            hard_preds = torch.argmax(predictions, dim=1).float()  # [B, H, W]
        else:  # Binary classification case
            hard_preds = (predictions[:, 0] > 0.5).float()  # [B, H, W]
            
        for i in range(batch_size):
            pred = hard_preds[i]  # [H, W]
            
            # Calculate consistency scores for various constraints
            continuity_score = self.check_continuity(pred)
            compactness_score = self.check_compactness(pred)
            smoothness_score = self.check_smoothness(pred)
            
            # Weighted combination of all constraint scores
            consistency_scores[i] = (
                self.continuity_weight * continuity_score +
                self.compactness_weight * compactness_score +
                self.smoothness_weight * smoothness_score
            ) / (self.continuity_weight + self.compactness_weight + self.smoothness_weight)
            
        return consistency_scores

    def check_continuity(self, pred):
        """Check continuity constraint of segmentation results"""
        # Apply closing operation (dilation followed by erosion) to fill small holes
        pred_expanded = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Dilation operation
        dilated = F.conv2d(
            pred_expanded, 
            self.dilate_kernel.unsqueeze(0).unsqueeze(0),
            padding=2
        )
        dilated = (dilated > 0).float()
        
        # Erosion operation
        closed = F.conv2d(
            dilated,
            self.erode_kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )
        closed = (closed >= 9).float()  # 3x3 kernel, result is 9 when all are 1
        
        # Calculate consistency between closed result and original segmentation
        intersection = (closed.squeeze() * pred).sum()
        union = closed.squeeze().sum() + pred.sum() - intersection
        if union == 0:
            return 1.0  # If both are empty, consider as perfectly consistent
        
        continuity_score = intersection / (union + 1e-6)
        return continuity_score.item()

    def check_compactness(self, pred):
        """Check compactness constraint of segmentation results"""
        # Calculate perimeter-to-area ratio as compactness measure
        pred_expanded = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Calculate edges
        edge = F.conv2d(
            pred_expanded,
            self.edge_kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )
        edge = (edge.abs() > 0.1).float()
        
        # Calculate perimeter (edge pixel count) and area
        perimeter = edge.sum().item()
        area = pred.sum().item()
        
        if area == 0:
            return 1.0  # If area is 0, return 1 for perfect compactness
            
        # Compactness: 4π·area/perimeter², closer to 1 means closer to circular shape
        compactness = (4 * 3.14159 * area) / (perimeter ** 2 + 1e-6)
        
        # Normalize to 0-1 range
        compactness = min(max(compactness, 0), 1)
        return compactness

    def check_smoothness(self, pred):
        """Check smoothness constraint of segmentation results"""
        pred_expanded = pred.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        
        # Apply smoothing convolution kernel
        smoothed = F.conv2d(
            pred_expanded,
            self.smooth_kernel.unsqueeze(0).unsqueeze(0),
            padding=1
        )
        
        # Calculate similarity between original prediction and smoothed result
        diff = ((smoothed - pred_expanded) ** 2).mean()
        smoothness_score = torch.exp(-10 * diff)  # Convert to 0-1 score, close to 1 when difference is small
        
        return smoothness_score.item()

    def perform_abduction(self, predictions, mask=None):
        """
        Perform abductive reasoning to correct prediction results
        
        Args:
            predictions: Model prediction results [B, C, H, W]
            mask: Correction mask [B, 1, H, W], indicating which regions need correction
            
        Returns:
            Corrected prediction results [B, C, H, W]
        """
        if mask is None:
            # If no mask is provided, use all-ones mask (correct all regions)
            mask = torch.ones_like(predictions[:, :1])
            
        batch_size = predictions.shape[0]
        num_classes = predictions.shape[1]
        corrected_preds = predictions.clone()
        
        for i in range(batch_size):
            pred = predictions[i]  # [C, H, W]
            m = mask[i]  # [1, H, W]
            
            # Special handling for binary classification case
            if num_classes == 1:
                # Perform morphological correction
                pred_expanded = pred.unsqueeze(0)  # [1, 1, H, W]
                
                # First apply closing operation (dilation followed by erosion) to fill small holes
                dilated = F.conv2d(
                    pred_expanded, 
                    self.dilate_kernel.unsqueeze(0).unsqueeze(0),
                    padding=2
                )
                dilated = (dilated > 0).float()
                
                closed = F.conv2d(
                    dilated,
                    self.erode_kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                )
                closed = (closed >= 9).float()
                
                # Then apply opening operation (erosion followed by dilation) to remove small protrusions
                eroded = F.conv2d(
                    closed,
                    self.erode_kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                )
                eroded = (eroded >= 9).float()
                
                opened = F.conv2d(
                    eroded,
                    self.dilate_kernel.unsqueeze(0).unsqueeze(0),
                    padding=2
                )
                opened = (opened > 0).float()
                
                # Apply smoothing operation
                smoothed = F.conv2d(
                    opened,
                    self.smooth_kernel.unsqueeze(0).unsqueeze(0),
                    padding=1
                )
                
                # Only apply correction in regions specified by mask
                final_pred = pred_expanded * m + smoothed * (1 - m)
                corrected_preds[i] = final_pred.squeeze(0)
                
            else:
                # For multi-class case, process each class separately
                hard_pred = torch.argmax(pred, dim=0).float()  # [H, W]
                
                # Perform same morphological correction for each class
                for c in range(num_classes):
                    class_mask = (hard_pred == c).float()
                    class_mask_expanded = class_mask.unsqueeze(0).unsqueeze(0)
                    
                    # Perform morphological closing and opening operations
                    dilated = F.conv2d(
                        class_mask_expanded, 
                        self.dilate_kernel.unsqueeze(0).unsqueeze(0),
                        padding=2
                    )
                    dilated = (dilated > 0).float()
                    
                    closed = F.conv2d(
                        dilated,
                        self.erode_kernel.unsqueeze(0).unsqueeze(0),
                        padding=1
                    )
                    closed = (closed >= 9).float()
                    
                    eroded = F.conv2d(
                        closed,
                        self.erode_kernel.unsqueeze(0).unsqueeze(0),
                        padding=1
                    )
                    eroded = (eroded >= 9).float()
                    
                    opened = F.conv2d(
                        eroded,
                        self.dilate_kernel.unsqueeze(0).unsqueeze(0),
                        padding=2
                    )
                    opened = (opened > 0).float()
                    
                    # Apply smoothing operation
                    smoothed = F.conv2d(
                        opened,
                        self.smooth_kernel.unsqueeze(0).unsqueeze(0),
                        padding=1
                    ).squeeze()
                    
                    # Only apply correction in regions specified by mask and current class region
                    class_correction = smoothed * (1 - m.squeeze())
                    
                    # Update prediction probability for this class
                    corrected_preds[i, c] = corrected_preds[i, c] * m.squeeze() + class_correction
                    
                # Ensure sum equals 1 for multi-class case
                if num_classes > 1:
                    corrected_preds[i] = F.softmax(corrected_preds[i], dim=0)
                    
        return corrected_preds
