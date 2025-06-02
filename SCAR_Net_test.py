import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from models.scar_net import create_scar_net
from utils.dataset import SCARDataset
from utils.transforms import get_transforms
from utils.metrics import (
    calculate_segmentation_metrics,
    calculate_classification_metrics,
    MetricTracker
)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SCAR-Net Testing')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data/test',
                        help='Directory containing the test dataset')
    parser.add_argument('--img_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='Number of input channels')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Number of output segmentation classes')

    # Model parameters
    parser.add_argument('--model_type', type=str, default='standard',
                        choices=['standard', 'lite', 'custom'],
                        help='Type of SCAR-Net model to use')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Test parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for testing')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Directory to save test results')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save visualizations of predictions')
    parser.add_argument('--save_metrics', action='store_true',
                        help='Save metrics to CSV file')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for segmentation predictions')

    return parser.parse_args()


def load_model_from_checkpoint(checkpoint_path, device):
    """
    Load model from checkpoint

    Args:
        checkpoint_path: Path to model checkpoint
        device: Device to load model to

    Returns:
        model: Loaded model
        args: Arguments used to create the model
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get arguments used to create the model
    args = checkpoint['args']

    # Create model
    model = create_scar_net(
        model_type=args.model_type,
        img_size=args.img_size,
        patch_size=args.patch_size,
        in_chans=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        window_size=args.window_size
    )

    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, args


def visualize_results(images, seg_masks, seg_preds, case_ids, output_dir, threshold=0.5):
    """
    Visualize and save segmentation results

    Args:
        images: Batch of images
        seg_masks: Batch of ground truth segmentation masks
        seg_preds: Batch of predicted segmentation masks
        case_ids: Batch of case IDs
        output_dir: Directory to save visualizations
        threshold: Threshold for segmentation predictions
    """
    os.makedirs(output_dir, exist_ok=True)

    # Move tensors to CPU and convert to numpy
    images = images.cpu().numpy()
    seg_masks = seg_masks.cpu().numpy()
    seg_preds = seg_preds.cpu().numpy()

    # Apply threshold to predictions
    seg_preds_binary = (seg_preds > threshold).astype(np.float32)

    # Process each image in the batch
    for i in range(images.shape[0]):
        # Create figure with 3 subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        # Display image
        image = np.transpose(images[i], (1, 2, 0))  # CHW -> HWC
        image = (image - image.min()) / (image.max() - image.min())  # Normalize to [0, 1]
        axs[0].imshow(image)
        axs[0].set_title('Input Image')
        axs[0].axis('off')

        # Display ground truth mask
        axs[1].imshow(image)
        mask = seg_masks[i, 0]  # First channel if multi-channel
        axs[1].imshow(mask, alpha=0.5, cmap='jet')
        axs[1].set_title('Ground Truth')
        axs[1].axis('off')

        # Display predicted mask
        axs[2].imshow(image)
        pred = seg_preds_binary[i, 0]  # First channel if multi-channel
        axs[2].imshow(pred, alpha=0.5, cmap='jet')
        axs[2].set_title('Prediction')
        axs[2].axis('off')

        # Save figure
        case_id = case_ids[i]
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'case_{case_id}_visualization.png'), dpi=200)
        plt.close(fig)


def test(model, test_loader, device, args):
    """
    Test model on test set

    Args:
        model: The SCAR-Net model
        test_loader: DataLoader for test set
        device: Device to use (cuda or cpu)
        args: Command line arguments

    Returns:
        dict: Dictionary with test metrics
    """
    model.eval()
    metric_tracker = MetricTracker()

    # Lists to store results for each case
    all_case_ids = []
    all_scar_preds = []
    all_recurrence_preds = []
    all_scar_labels = []
    all_recurrence_labels = []
    all_dice_scores = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            # Get data
            images = batch['image'].to(device)
            seg_masks = batch['segmentation'].to(device)
            scar_labels = batch['scar_label'].to(device)
            recurrence_labels = batch['recurrence_label'].to(device)
            case_ids = batch['case_id']

            # Forward pass
            outputs = model(images)

            # Apply sigmoid for segmentation
            seg_preds = torch.sigmoid(outputs['segmentation'])
            seg_preds_binary = (seg_preds > args.threshold).float()

            # Get binary predictions for classification
            scar_preds = torch.sigmoid(outputs['scar_prob'])
            recurrence_preds = torch.sigmoid(outputs['recurrence_prob'])

            # Update metric tracker
            metric_tracker.update(
                seg_preds_binary.cpu(), seg_masks.cpu(),
                scar_preds.cpu(), scar_labels.cpu(),
                recurrence_preds.cpu(), recurrence_labels.cpu()
            )

            # Save case-wise results
            all_case_ids.extend(case_ids)
            all_scar_preds.extend(scar_preds.cpu().numpy().flatten())
            all_recurrence_preds.extend(recurrence_preds.cpu().numpy().flatten())
            all_scar_labels.extend(scar_labels.cpu().numpy().flatten())
            all_recurrence_labels.extend(recurrence_labels.cpu().numpy().flatten())

            # Calculate Dice score for each case
            for i in range(seg_preds_binary.size(0)):
                dice = calculate_segmentation_metrics(
                    seg_preds_binary[i].unsqueeze(0).cpu(),
                    seg_masks[i].unsqueeze(0).cpu(),
                    threshold=args.threshold
                )['dice']
                all_dice_scores.append(dice)

            # Save visualizations if requested
            if args.save_visualizations:
                visualize_results(
                    images, seg_masks, seg_preds,
                    case_ids, os.path.join(args.output_dir, 'visualizations'),
                    threshold=args.threshold
                )

    # Get aggregated metrics
    metrics = metric_tracker.get_results()

    # Create case-wise results dataframe
    results_df = pd.DataFrame({
        'case_id': all_case_ids,
        'scar_pred': all_scar_preds,
        'scar_label': all_scar_labels,
        'recurrence_pred': all_recurrence_preds,
        'recurrence_label': all_recurrence_labels,
        'dice_score': all_dice_scores
    })

    # Save case-wise results if requested
    if args.save_metrics:
        os.makedirs(args.output_dir, exist_ok=True)
        results_df.to_csv(os.path.join(args.output_dir, 'case_wise_results.csv'), index=False)

        # Save aggregated metrics
        metrics_df = pd.DataFrame({k: [v] for k, v in metrics.items()})
        metrics_df.to_csv(os.path.join(args.output_dir, 'aggregated_metrics.csv'), index=False)

    return metrics, results_df


def print_metrics(metrics):
    """
    Print metrics in a formatted way

    Args:
        metrics: Dictionary with metrics
    """
    print("\n===== Test Results =====")

    print("\nSegmentation Metrics:")
    for key, value in metrics.items():
        if key.startswith('seg_'):
            print(f"  {key[4:]}: {value:.4f}")

    print("\nScar Classification Metrics:")
    for key, value in metrics.items():
        if key.startswith('scar_'):
            print(f"  {key[5:]}: {value:.4f}")

    print("\nRecurrence Classification Metrics:")
    for key, value in metrics.items():
        if key.startswith('recurrence_'):
            print(f"  {key[11:]}: {value:.4f}")


def create_confusion_matrix_plot(results_df, output_dir):
    """
    Create and save confusion matrix plots for scar and recurrence classification

    Args:
        results_df: DataFrame with case-wise results
        output_dir: Directory to save plots
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    os.makedirs(output_dir, exist_ok=True)

    # Create scar classification confusion matrix
    scar_pred = (results_df['scar_pred'] > 0.5).astype(int)
    scar_true = results_df['scar_label'].astype(int)
    scar_cm = confusion_matrix(scar_true, scar_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(scar_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Scar', 'Scar'],
                yticklabels=['No Scar', 'Scar'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Scar Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scar_confusion_matrix.png'), dpi=200)
    plt.close()

    # Create recurrence classification confusion matrix
    recurrence_pred = (results_df['recurrence_pred'] > 0.5).astype(int)
    recurrence_true = results_df['recurrence_label'].astype(int)
    recurrence_cm = confusion_matrix(recurrence_true, recurrence_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(recurrence_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Recurrence', 'Recurrence'],
                yticklabels=['No Recurrence', 'Recurrence'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Recurrence Classification Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recurrence_confusion_matrix.png'), dpi=200)
    plt.close()


def create_roc_curves(results_df, output_dir):
    """
    Create and save ROC curves for scar and recurrence classification

    Args:
        results_df: DataFrame with case-wise results
        output_dir: Directory to save plots
    """
    from sklearn.metrics import roc_curve, auc

    os.makedirs(output_dir, exist_ok=True)

    # Create scar classification ROC curve
    fpr, tpr, _ = roc_curve(results_df['scar_label'], results_df['scar_pred'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Scar Classification ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'scar_roc_curve.png'), dpi=200)
    plt.close()

    # Create recurrence classification ROC curve
    fpr, tpr, _ = roc_curve(results_df['recurrence_label'], results_df['recurrence_pred'])
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Recurrence Classification ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'recurrence_roc_curve.png'), dpi=200)
    plt.close()


def create_dice_distribution_plot(results_df, output_dir):
    """
    Create and save a histogram of Dice scores

    Args:
        results_df: DataFrame with case-wise results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.hist(results_df['dice_score'], bins=20, color='steelblue', edgecolor='black')
    plt.axvline(results_df['dice_score'].mean(), color='red', linestyle='dashed', linewidth=2,
                label=f'Mean Dice: {results_df["dice_score"].mean():.4f}')
    plt.axvline(results_df['dice_score'].median(), color='green', linestyle='dashed', linewidth=2,
                label=f'Median Dice: {results_df["dice_score"].median():.4f}')
    plt.xlabel('Dice Score')
    plt.ylabel('Number of Cases')
    plt.title('Distribution of Dice Scores')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dice_distribution.png'), dpi=200)
    plt.close()


def main():
    # Get arguments
    args = get_args()

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model, model_args = load_model_from_checkpoint(args.checkpoint, device)

    # Use model arguments for image size if not specified
    if not hasattr(args, 'img_size') or args.img_size is None:
        args.img_size = model_args.img_size

    # Set up transforms
    _, test_transform = get_transforms(args.img_size)

    # Create test dataset
    test_dataset = SCARDataset(
        root_dir=args.data_dir,
        transform=test_transform,
        img_size=args.img_size
    )

    # Create test data loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Testing on {len(test_dataset)} samples")

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    args.output_dir = os.path.join(args.output_dir, f"test_{timestamp}")
    os.makedirs(args.output_dir, exist_ok=True)

    # Test model
    metrics, results_df = test(model, test_loader, device, args)

    # Print metrics
    print_metrics(metrics)

    # Create and save additional plots
    print("Creating visualizations...")
    create_confusion_matrix_plot(results_df, args.output_dir)
    create_roc_curves(results_df, args.output_dir)
    create_dice_distribution_plot(results_df, args.output_dir)

    print(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
