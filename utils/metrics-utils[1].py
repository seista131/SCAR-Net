import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def dice_coefficient(pred, target, smooth=1e-6):
    """
    Calculate Dice coefficient

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        float: Dice coefficient
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate Dice coefficient
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat)

    return (2.0 * intersection + smooth) / (union + smooth)


def iou_score(pred, target, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) score

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        float: IoU score
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate IoU
    intersection = torch.sum(pred_flat * target_flat)
    union = torch.sum(pred_flat) + torch.sum(target_flat) - intersection

    return (intersection + smooth) / (union + smooth)


def pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask

    Returns:
        float: Pixel accuracy
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate accuracy
    correct = torch.sum(pred_flat == target_flat)
    total = pred_flat.size(0)

    return correct.float() / total


def sensitivity(pred, target, smooth=1e-6):
    """
    Calculate sensitivity (recall or true positive rate)

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        float: Sensitivity
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate true positives and false negatives
    tp = torch.sum(pred_flat * target_flat)
    fn = torch.sum((1 - pred_flat) * target_flat)

    return (tp + smooth) / (tp + fn + smooth)


def specificity(pred, target, smooth=1e-6):
    """
    Calculate specificity (true negative rate)

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        float: Specificity
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate true negatives and false positives
    tn = torch.sum((1 - pred_flat) * (1 - target_flat))
    fp = torch.sum(pred_flat * (1 - target_flat))

    return (tn + smooth) / (tn + fp + smooth)


def precision(pred, target, smooth=1e-6):
    """
    Calculate precision (positive predictive value)

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        smooth: Smoothing factor to avoid division by zero

    Returns:
        float: Precision
    """
    # Ensure inputs are binary
    pred = pred.float()
    target = target.float()

    # Flatten the arrays
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)

    # Calculate true positives and false positives
    tp = torch.sum(pred_flat * target_flat)
    fp = torch.sum(pred_flat * (1 - target_flat))

    return (tp + smooth) / (tp + fp + smooth)


def hausdorff_distance(pred, target):
    """
    Calculate Hausdorff distance between predicted and target masks

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask

    Returns:
        float: Hausdorff distance
    """
    from scipy.ndimage import distance_transform_edt

    # Convert to numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure inputs are binary
    pred = (pred > 0.5).astype(np.bool)
    target = (target > 0.5).astype(np.bool)

    # If either mask is empty, return maximum distance
    if not np.any(pred) or not np.any(target):
        return np.sqrt(pred.shape[0] ** 2 + pred.shape[1] ** 2)

    # Calculate distance transforms
    dt_pred = distance_transform_edt(~pred)
    dt_target = distance_transform_edt(~target)

    # Calculate Hausdorff distance
    hausdorff_dt = np.maximum(np.max(dt_pred[target]), np.max(dt_target[pred]))

    return hausdorff_dt


def surface_dice(pred, target, tolerance=1):
    """
    Calculate Surface Dice at the specified tolerance level

    Args:
        pred: Predicted binary mask (after sigmoid/thresholding)
        target: Target binary mask
        tolerance: Distance tolerance in pixels

    Returns:
        float: Surface Dice score
    """
    from scipy.ndimage import distance_transform_edt

    # Convert to numpy arrays
    if isinstance(pred, torch.Tensor):
        pred = pred.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    # Ensure inputs are binary
    pred = (pred > 0.5).astype(np.bool)
    target = (target > 0.5).astype(np.bool)

    # If either mask is empty, return 0
    if not np.any(pred) and not np.any(target):
        return 1.0  # Both empty, perfect match
    elif not np.any(pred) or not np.any(target):
        return 0.0  # One is empty, the other isn't

    # Get the surface voxels of each mask
    surface_pred = get_surface(pred)
    surface_target = get_surface(target)

    # Calculate distance maps
    dt_pred = distance_transform_edt(~surface_pred)
    dt_target = distance_transform_edt(~surface_target)

    # Count surface voxels within tolerance
    pred_in_tolerance = np.sum(dt_target[surface_pred] <= tolerance)
    target_in_tolerance = np.sum(dt_pred[surface_target] <= tolerance)

    # Calculate Surface Dice
    surface_dice = (pred_in_tolerance + target_in_tolerance) / (np.sum(surface_pred) + np.sum(surface_target))

    return surface_dice


def get_surface(binary_img):
    """Helper function to get the surface voxels of a binary mask"""
    from scipy.ndimage import binary_erosion

    # Erode the binary image to get the interior
    eroded = binary_erosion(binary_img)

    # Surface is the difference between original and eroded
    surface = np.logical_xor(binary_img, eroded)

    return surface


def calculate_classification_metrics(preds, targets):
    """
    Calculate classification metrics

    Args:
        preds: Predicted probabilities or binary labels
        targets: Target binary labels

    Returns:
        dict: Dictionary containing classification metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Ensure inputs are 1D arrays
    preds = preds.flatten()
    targets = targets.flatten()

    # Get binary predictions
    binary_preds = (preds > 0.5).astype(int)

    # Calculate metrics
    acc = accuracy_score(targets, binary_preds)
    prec = precision_score(targets, binary_preds, zero_division=0)
    rec = recall_score(targets, binary_preds, zero_division=0)
    f1 = f1_score(targets, binary_preds, zero_division=0)

    # Calculate AUC if predictions are probabilities
    if np.any((preds > 0) & (preds < 1)):
        auc = roc_auc_score(targets, preds)
    else:
        auc = 0.5  # Default value for binary predictions

    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()

    # Calculate specificity
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Return metrics as dictionary
    return {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'specificity': spec,
        'f1_score': f1,
        'auc': auc,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }


def calculate_segmentation_metrics(preds, targets, threshold=0.5):
    """
    Calculate comprehensive segmentation metrics

    Args:
        preds: Predicted probabilities or binary masks
        targets: Target binary masks
        threshold: Threshold for converting probabilities to binary masks

    Returns:
        dict: Dictionary containing segmentation metrics
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    # Apply threshold to get binary masks
    binary_preds = (preds > threshold).astype(np.float32)
    binary_targets = (targets > threshold).astype(np.float32)

    # Convert back to tensors for some metrics
    pred_tensor = torch.tensor(binary_preds)
    target_tensor = torch.tensor(binary_targets)

    # Calculate metrics
    dice = dice_coefficient(pred_tensor, target_tensor).item()
    iou = iou_score(pred_tensor, target_tensor).item()
    pixel_acc = pixel_accuracy(pred_tensor, target_tensor).item()
    sens = sensitivity(pred_tensor, target_tensor).item()
    spec = specificity(pred_tensor, target_tensor).item()
    prec = precision(pred_tensor, target_tensor).item()

    # Calculate Hausdorff distance and Surface Dice for each slice
    hausdorff_dists = []
    surface_dices = []

    # Handle different dimensions
    if binary_preds.ndim == 4:  # Batch of 3D volumes (B, C, H, W)
        for b in range(binary_preds.shape[0]):
            for c in range(binary_preds.shape[1]):
                hausdorff_dists.append(hausdorff_distance(binary_preds[b, c], binary_targets[b, c]))
                surface_dices.append(surface_dice(binary_preds[b, c], binary_targets[b, c]))
    elif binary_preds.ndim == 3:  # Batch of 2D images (B, H, W)
        for b in range(binary_preds.shape[0]):
            hausdorff_dists.append(hausdorff_distance(binary_preds[b], binary_targets[b]))
            surface_dices.append(surface_dice(binary_preds[b], binary_targets[b]))
    else:  # Single 2D image (H, W)
        hausdorff_dists.append(hausdorff_distance(binary_preds, binary_targets))
        surface_dices.append(surface_dice(binary_preds, binary_targets))

    # Average the metrics
    avg_hausdorff = np.mean(hausdorff_dists) if hausdorff_dists else np.nan
    avg_surface_dice = np.mean(surface_dices) if surface_dices else np.nan

    # Return metrics as dictionary
    return {
        'dice': dice,
        'iou': iou,
        'pixel_accuracy': pixel_acc,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'hausdorff_distance': avg_hausdorff,
        'surface_dice': avg_surface_dice
    }


class MetricTracker:
    """
    Class to track and update metrics during training/evaluation
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics"""
        self.seg_metrics = {
            'dice': [],
            'iou': [],
            'pixel_accuracy': [],
            'sensitivity': [],
            'specificity': [],
            'precision': [],
            'hausdorff_distance': [],
            'surface_dice': []
        }

        self.scar_cls_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1_score': [],
            'auc': []
        }

        self.recurrence_cls_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1_score': [],
            'auc': []
        }

    def update(self, seg_preds, seg_targets, scar_preds, scar_targets, recurrence_preds, recurrence_targets):
        """
        Update metrics with new batch of predictions and targets

        Args:
            seg_preds: Segmentation predictions
            seg_targets: Segmentation targets
            scar_preds: Scar classification predictions
            scar_targets: Scar classification targets
            recurrence_preds: Recurrence classification predictions
            recurrence_targets: Recurrence classification targets
        """
        # Calculate segmentation metrics
        seg_metrics = calculate_segmentation_metrics(seg_preds, seg_targets)

        # Calculate classification metrics
        scar_metrics = calculate_classification_metrics(scar_preds, scar_targets)
        recurrence_metrics = calculate_classification_metrics(recurrence_preds, recurrence_targets)

        # Update tracked metrics
        for key, value in seg_metrics.items():
            self.seg_metrics[key].append(value)

        for key, value in scar_metrics.items():
            if key in self.scar_cls_metrics:
                self.scar_cls_metrics[key].append(value)

        for key, value in recurrence_metrics.items():
            if key in self.recurrence_cls_metrics:
                self.recurrence_cls_metrics[key].append(value)

    def get_results(self):
        """
        Get average metrics

        Returns:
            dict: Dictionary with average metrics
        """
        results = {}

        # Calculate averages for segmentation metrics
        for key, values in self.seg_metrics.items():
            if values:
                results[f'seg_{key}'] = np.mean(values)
            else:
                results[f'seg_{key}'] = np.nan

        # Calculate averages for scar classification metrics
        for key, values in self.scar_cls_metrics.items():
            if values:
                results[f'scar_{key}'] = np.mean(values)
            else:
                results[f'scar_{key}'] = np.nan

        # Calculate averages for recurrence classification metrics
        for key, values in self.recurrence_cls_metrics.items():
            if values:
                results[f'recurrence_{key}'] = np.mean(values)
            else:
                results[f'recurrence_{key}'] = np.nan

        return results