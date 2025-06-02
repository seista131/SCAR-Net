import argparse
import os
import random
import time
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from models.scar_net import create_scar_net
from utils.losses import CombinedLoss, MTLWeightedLoss
from utils.metrics import MetricTracker
from utils.dataset import SCARDataset
from utils.transforms import get_transforms


def set_seed(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SCAR-Net Training')

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Directory containing the dataset')
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
    parser.add_argument('--embed_dim', type=int, default=96,
                        help='Embedding dimension')
    parser.add_argument('--window_size', type=int, default=7,
                        help='Window size for Swin Transformer')
    parser.add_argument('--patch_size', type=int, default=4,
                        help='Patch size for patch embedding')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=8,
                        help='Batch size for validation')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'],
                        help='Learning rate scheduler')
    parser.add_argument('--loss_type', type=str, default='combined',
                        choices=['combined', 'weighted'],
                        help='Loss function type')
    parser.add_argument('--seg_weight', type=float, default=1.0,
                        help='Weight for segmentation loss')
    parser.add_argument('--scar_cls_weight', type=float, default=0.5,
                        help='Weight for scar classification loss')
    parser.add_argument('--recurrence_cls_weight', type=float, default=0.5,
                        help='Weight for recurrence classification loss')

    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save_dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Directory to save logs')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validation interval in epochs')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Checkpoint saving interval in epochs')

    return parser.parse_args()


def train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """
    Train model for one epoch

    Args:
        model: The SCAR-Net model
        train_loader: DataLoader for training set
        optimizer: Optimizer for training
        criterion: Loss function
        device: Device to use (cuda or cpu)
        epoch: Current epoch number

    Returns:
        dict: Dictionary with training metrics
    """
    model.train()
    epoch_loss = 0
    metric_tracker = MetricTracker()

    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        # Get data
        images = batch['image'].to(device)
        seg_masks = batch['segmentation'].to(device)
        scar_labels = batch['scar_label'].to(device)
        recurrence_labels = batch['recurrence_label'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Prepare targets
        targets = {
            'segmentation': seg_masks,
            'scar_label': scar_labels,
            'recurrence_label': recurrence_labels
        }

        # Calculate loss
        loss, loss_dict = criterion(outputs, targets)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Update epoch loss
        epoch_loss += loss.item()

        # Update metrics
        with torch.no_grad():
            # Apply sigmoid and threshold for segmentation
            seg_preds = torch.sigmoid(outputs['segmentation'])
            seg_preds_binary = (seg_preds > 0.5).float()

            # Get binary predictions for classification
            scar_preds = torch.sigmoid(outputs['scar_prob'])
            recurrence_preds = torch.sigmoid(outputs['recurrence_prob'])

            # Update metric tracker
            metric_tracker.update(
                seg_preds_binary.cpu(), seg_masks.cpu(),
                scar_preds.cpu(), scar_labels.cpu(),
                recurrence_preds.cpu(), recurrence_labels.cpu()
            )

        # Print progress
        if (batch_idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch} | Batch {batch_idx + 1}/{len(train_loader)} | "
                  f"Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")

    # Calculate average epoch loss
    epoch_loss /= len(train_loader)

    # Get metrics
    metrics = metric_tracker.get_results()
    metrics['loss'] = epoch_loss

    return metrics


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set

    Args:
        model: The SCAR-Net model
        val_loader: DataLoader for validation set
        criterion: Loss function
        device: Device to use (cuda or cpu)

    Returns:
        dict: Dictionary with validation metrics
    """
    model.eval()
    val_loss = 0
    metric_tracker = MetricTracker()

    with torch.no_grad():
        for batch in val_loader:
            # Get data
            images = batch['image'].to(device)
            seg_masks = batch['segmentation'].to(device)
            scar_labels = batch['scar_label'].to(device)
            recurrence_labels = batch['recurrence_label'].to(device)

            # Forward pass
            outputs = model(images)

            # Prepare targets
            targets = {
                'segmentation': seg_masks,
                'scar_label': scar_labels,
                'recurrence_label': recurrence_labels
            }

            # Calculate loss
            loss, _ = criterion(outputs, targets)

            # Update validation loss
            val_loss += loss.item()

            # Apply sigmoid and threshold for segmentation
            seg_preds = torch.sigmoid(outputs['segmentation'])
            seg_preds_binary = (seg_preds > 0.5).float()

            # Get binary predictions for classification
            scar_preds = torch.sigmoid(outputs['scar_prob'])
            recurrence_preds = torch.sigmoid(outputs['recurrence_prob'])

            # Update metric tracker
            metric_tracker.update(
                seg_preds_binary.cpu(), seg_masks.cpu(),
                scar_preds.cpu(), scar_labels.cpu(),
                recurrence_preds.cpu(), recurrence_labels.cpu()
            )

    # Calculate average validation loss
    val_loss /= len(val_loader)

    # Get metrics
    metrics = metric_tracker.get_results()
    metrics['loss'] = val_loss

    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args, is_best=False):
    """Save model checkpoint"""
    os.makedirs(args.save_dir, exist_ok=True)

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_metrics': val_metrics,
        'args': args
    }

    # Save regular checkpoint
    if epoch % args.save_interval == 0:
        checkpoint_path = os.path.join(args.save_dir, f'scar_net_epoch_{epoch}.pth')
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save best model
    if is_best:
        best_path = os.path.join(args.save_dir, 'scar_net_best.pth')
        torch.save(checkpoint, best_path)
        print(f"Best model saved to {best_path}")


def log_metrics(writer, metrics, epoch, prefix='train'):
    """Log metrics to TensorBoard"""
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, epoch)


def main():
    # Get arguments
    args = get_args()

    # Set random seed
    set_seed(args.seed)

    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up transforms
    train_transform, val_transform = get_transforms(args.img_size)

    # Create datasets
    full_dataset = SCARDataset(
        root_dir=args.data_dir,
        transform=None,
        img_size=args.img_size
    )

    # Split dataset into train and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training on {train_size} samples, validating on {val_size} samples")

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
    model = model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Create learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    else:  # 'none'
        scheduler = None

    # Create loss function
    if args.loss_type == 'combined':
        criterion = CombinedLoss(
            seg_loss_weight=args.seg_weight,
            scar_cls_weight=args.scar_cls_weight,
            recurrence_cls_weight=args.recurrence_cls_weight
        )
    else:  # 'weighted'
        criterion = MTLWeightedLoss()

    # Set up TensorBoard writer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_dir, f"{args.model_type}_{timestamp}")
    writer = SummaryWriter(log_dir)

    # Log model architecture
    dummy_input = torch.zeros(1, args.in_channels, args.img_size, args.img_size).to(device)
    writer.add_graph(model, dummy_input)

    # Training loop
    best_val_dice = 0

    print(f"Starting training for {args.epochs} epochs")

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch)

        # Log training metrics
        log_metrics(writer, train_metrics, epoch, prefix='train')

        # Print training metrics
        print("Training metrics:")
        for key, value in train_metrics.items():
            if key in ['loss', 'seg_dice', 'scar_accuracy', 'recurrence_accuracy']:
                print(f"  {key}: {value:.4f}")

        # Validate
        if epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, criterion, device)

            # Log validation metrics
            log_metrics(writer, val_metrics, epoch, prefix='val')

            # Print validation metrics
            print("Validation metrics:")
            for key, value in val_metrics.items():
                if key in ['loss', 'seg_dice', 'scar_accuracy', 'recurrence_accuracy']:
                    print(f"  {key}: {value:.4f}")

            # Update learning rate scheduler
            if args.scheduler == 'plateau':
                scheduler.step(val_metrics['loss'])

            # Check if this is the best model
            is_best = val_metrics['seg_dice'] > best_val_dice
            if is_best:
                best_val_dice = val_metrics['seg_dice']
                print(f"New best validation Dice: {best_val_dice:.4f}")

            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, val_metrics, args, is_best)

        # Update learning rate scheduler
        if args.scheduler == 'cosine':
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar('train/learning_rate', current_lr, epoch)

    # Close TensorBoard writer
    writer.close()

    print("Training completed!")


if __name__ == '__main__':
    main()
