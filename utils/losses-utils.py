import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation tasks

    Computes the Sørensen-Dice loss between the predicted and target masks.
    The Dice coefficient ranges from 0 to 1, where 1 means perfect overlap.
    This loss is 1 - Dice coefficient.
    """

    def __init__(self, smooth=1.0, squared_pred=False, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.squared_pred = squared_pred
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted segmentation mask (B, C, H, W) or (B, H, W)
            target: Target segmentation mask (B, C, H, W) or (B, H, W)

        Returns:
            loss: Dice loss
        """
        # Ensure the predictions are probabilities
        if not self.squared_pred:
            pred = torch.sigmoid(pred)

        # Flatten the tensors
        if pred.dim() == 4:  # (B, C, H, W)
            # Multi-class case
            B, C, H, W = pred.shape
            pred = pred.view(B, C, -1)
            target = target.view(B, C, -1)

            # Calculate Dice coefficient for each class
            intersection = torch.sum(pred * target, dim=2)
            union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            # Average over classes and batch
            if self.reduction == 'mean':
                return 1.0 - torch.mean(dice)
            elif self.reduction == 'sum':
                return C - torch.sum(dice)
            else:  # 'none'
                return 1.0 - dice.mean(dim=1)
        else:  # (B, H, W)
            # Binary case
            pred = pred.view(-1)
            target = target.view(-1)

            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target)
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)

            return 1.0 - dice


class BinaryFocalLoss(nn.Module):
    """
    Binary Focal Loss for classification tasks

    Focal Loss addresses class imbalance by down-weighting easy examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted probability (B, 1) or (B,)
            target: Target label (B, 1) or (B,)

        Returns:
            loss: Focal loss
        """
        # Ensure pred and target have same shape
        pred = pred.view(-1)
        target = target.view(-1)

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')

        # Apply sigmoid to get probabilities
        pred_prob = torch.sigmoid(pred)

        # Calculate focal weight
        p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
        alpha_t = target * self.alpha + (1 - target) * (1 - self.alpha)
        focal_weight = alpha_t * (1 - p_t).pow(self.gamma)

        # Apply weight to BCE loss
        loss = focal_weight * bce

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class CombinedLoss(nn.Module):
    """
    Combined loss for multi-task learning (segmentation + classification)
    """

    def __init__(
            self,
            seg_loss_weight=1.0,
            scar_cls_weight=0.5,
            recurrence_cls_weight=0.5,
            dice_smooth=1.0,
            focal_alpha=0.25,
            focal_gamma=2.0,
            reduction='mean'
    ):
        super(CombinedLoss, self).__init__()
        self.seg_loss_weight = seg_loss_weight
        self.scar_cls_weight = scar_cls_weight
        self.recurrence_cls_weight = recurrence_cls_weight
        self.reduction = reduction

        # Define individual loss functions
        self.dice_loss = DiceLoss(smooth=dice_smooth, reduction=reduction)
        self.focal_loss = BinaryFocalLoss(alpha=focal_alpha, gamma=focal_gamma, reduction=reduction)

    def forward(self, pred, target):
        """
        Args:
            pred: Dictionary containing:
                'segmentation': Segmentation prediction (B, 1, H, W)
                'scar_prob': Scar presence probability (B, 1)
                'recurrence_prob': Recurrence probability (B, 1)
            target: Dictionary containing:
                'segmentation': Target segmentation mask (B, 1, H, W)
                'scar_label': Scar presence label (B, 1)
                'recurrence_label': Recurrence label (B, 1)

        Returns:
            loss: Combined loss value
            loss_dict: Dictionary with individual loss values
        """
        # Segmentation loss
        seg_loss = self.dice_loss(pred['segmentation'], target['segmentation'])

        # Classification losses
        scar_cls_loss = self.focal_loss(pred['scar_prob'], target['scar_label'])
        recurrence_cls_loss = self.focal_loss(pred['recurrence_prob'], target['recurrence_label'])

        # Combined loss
        combined_loss = (
                self.seg_loss_weight * seg_loss +
                self.scar_cls_weight * scar_cls_loss +
                self.recurrence_cls_weight * recurrence_cls_loss
        )

        # Create loss dictionary
        loss_dict = {
            'segmentation_loss': seg_loss.item(),
            'scar_classification_loss': scar_cls_loss.item(),
            'recurrence_classification_loss': recurrence_cls_loss.item(),
            'combined_loss': combined_loss.item()
        }

        return combined_loss, loss_dict


class SoftDiceLoss(nn.Module):
    """
    Soft Dice Loss that works with raw logits
    """

    def __init__(self, smooth=1.0, reduction='mean'):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate Dice score
        intersection = torch.sum(pred_flat * target_flat)
        union = torch.sum(pred_flat) + torch.sum(target_flat)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)

        # Return Dice loss
        return 1.0 - dice_score


class TverskyLoss(nn.Module):
    """
    Tversky Loss - generalized Dice loss with weighting of FP and FN
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0, reduction='mean'):
        """
        Args:
            alpha: Weight for false positives
            beta: Weight for false negatives
            smooth: Smoothing factor
            reduction: Reduction method
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, pred, target):
        # Apply sigmoid to convert logits to probabilities
        pred = torch.sigmoid(pred)

        # Flatten the tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(pred_flat * target_flat)
        fp = torch.sum(pred_flat * (1 - target_flat))
        fn = torch.sum((1 - pred_flat) * target_flat)

        # Calculate Tversky index
        tversky_score = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Return Tversky loss
        return 1.0 - tversky_score


class BCEDiceLoss(nn.Module):
    """
    Combination of Binary Cross Entropy and Dice Loss
    """

    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1.0, reduction='mean'):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)
        self.dice_loss = SoftDiceLoss(smooth=smooth, reduction=reduction)

    def forward(self, pred, target):
        bce = self.bce_loss(pred, target)
        dice = self.dice_loss(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class MTLWeightedLoss(nn.Module):
    """
    Multi-Task Learning Weighted Loss with automatic weight balancing
    based on task uncertainty (Kendall et al. 2018)
    """

    def __init__(self, reduction='mean'):
        super(MTLWeightedLoss, self).__init__()
        self.reduction = reduction

        # Initialize log variance parameters for each task
        self.log_var_seg = nn.Parameter(torch.zeros(1))
        self.log_var_scar = nn.Parameter(torch.zeros(1))
        self.log_var_recurrence = nn.Parameter(torch.zeros(1))

        # Define loss functions
        self.dice_loss = SoftDiceLoss(reduction=reduction)
        self.bce_loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, pred, target):
        # Compute task losses
        seg_loss = self.dice_loss(pred['segmentation'], target['segmentation'])
        scar_loss = self.bce_loss(pred['scar_prob'], target['scar_label'])
        recurrence_loss = self.bce_loss(pred['recurrence_prob'], target['recurrence_label'])

        # Compute precision (inverse of variance) terms
        precision_seg = torch.exp(-self.log_var_seg)
        precision_scar = torch.exp(-self.log_var_scar)
        precision_recurrence = torch.exp(-self.log_var_recurrence)

        # Compute weighted losses
        weighted_seg_loss = precision_seg * seg_loss + self.log_var_seg
        weighted_scar_loss = precision_scar * scar_loss + self.log_var_scar
        weighted_recurrence_loss = precision_recurrence * recurrence_loss + self.log_var_recurrence

        # Compute total loss
        total_loss = weighted_seg_loss + weighted_scar_loss + weighted_recurrence_loss

        # Create loss dictionary
        loss_dict = {
            'segmentation_loss': seg_loss.item(),
            'scar_classification_loss': scar_loss.item(),
            'recurrence_classification_loss': recurrence_loss.item(),
            'weighted_segmentation_loss': weighted_seg_loss.item(),
            'weighted_scar_classification_loss': weighted_scar_loss.item(),
            'weighted_recurrence_classification_loss': weighted_recurrence_loss.item(),
            'combined_loss': total_loss.item(),
            'seg_weight': precision_seg.item(),
            'scar_weight': precision_scar.item(),
            'recurrence_weight': precision_recurrence.item()
        }

        return total_loss, loss_dict
