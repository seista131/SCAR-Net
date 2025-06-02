import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SobelFilter(nn.Module):
    """
    Sobel filter for calculating spatial gradients
    """

    def __init__(self):
        super(SobelFilter, self).__init__()

        # Define Sobel kernels for x and y directions
        self.sobel_x = torch.tensor([[-1, 0, 1],
                                     [-2, 0, 2],
                                     [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1],
                                     [0, 0, 0],
                                     [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)

        # Register buffers (non-trainable parameters)
        self.register_buffer('sobel_x_kernel', self.sobel_x)
        self.register_buffer('sobel_y_kernel', self.sobel_y)

    def forward(self, x):
        """
        Apply Sobel filters to input feature map to get gradients in x and y directions

        Args:
            x (tensor): Input feature map [B, C, H, W]

        Returns:
            tuple: Gradients in x and y directions [B, C, H, W]
        """
        # Get batch and channel dimensions
        B, C, H, W = x.shape

        # Reshape for channel-wise convolution
        x_reshaped = x.reshape(B * C, 1, H, W)

        # Apply Sobel filters
        grad_x = F.conv2d(x_reshaped, self.sobel_x_kernel, padding=1)
        grad_y = F.conv2d(x_reshaped, self.sobel_y_kernel, padding=1)

        # Reshape back to original dimensions
        grad_x = grad_x.reshape(B, C, H, W)
        grad_y = grad_y.reshape(B, C, H, W)

        return grad_x, grad_y


class SqueezeExcitation(nn.Module):
    """
    Squeeze and Excitation block for channel attention
    """

    def __init__(self, channel, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BSAN(nn.Module):
    """
    Boundary Sensitive Attention Network (BSAN)
    """

    def __init__(self, in_channels):
        super(BSAN, self).__init__()

        # Spatial gradient calculation
        self.sobel_filter = SobelFilter()

        # Gradient feature extraction
        self.gradient_extraction = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Gradient-feature fusion
        self.gradient_fusion = nn.Sequential(
            nn.Conv2d(64, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels)
        )

        # Feature refinement with SE block
        self.se_block = SqueezeExcitation(in_channels)

        # Activation for attention map
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of the BSAN module

        Args:
            x (tensor): Input feature map [B, C, H, W]

        Returns:
            tensor: Enhanced feature map with boundary attention [B, C, H, W]
        """
        # Store original input for residual connection
        identity = x

        # Convert to NCHW format if input is in NHWC format
        if x.dim() == 4 and x.shape[1] != x.shape[3]:  # NHWC format
            x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
            was_nhwc = True
        else:
            was_nhwc = False

        # Calculate spatial gradients
        grad_x, grad_y = self.sobel_filter(x)

        # Calculate gradient magnitude
        grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        # Average across channels to get a single gradient map
        grad_magnitude = torch.mean(grad_magnitude, dim=1, keepdim=True)

        # Extract features from gradient magnitude
        grad_features = self.gradient_extraction(grad_magnitude)

        # Fuse gradient features to match input channels
        grad_attention = self.gradient_fusion(grad_features)

        # Apply sigmoid to get attention weights
        attention_weights = self.sigmoid(grad_attention)

        # Apply attention to input features
        enhanced_features = x * attention_weights

        # Apply channel attention with SE block
        refined_features = self.se_block(enhanced_features)

        # Add residual connection
        output = refined_features + identity

        # Convert back to original format if needed
        if was_nhwc:
            output = output.permute(0, 2, 3, 1)  # NCHW -> NHWC

        return output


class BSANForSwinTransformer(nn.Module):
    """
    BSAN adapted specifically for Swin Transformer architecture
    Handles input in NHWC format (B, H*W, C) that Swin Transformer uses
    """

    def __init__(self, dim, input_resolution):
        super(BSANForSwinTransformer, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # Core BSAN module
        self.bsan = BSAN(dim)

        # Normalization layer
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Forward pass

        Args:
            x (tensor): Input feature map in NHWC format [B, H*W, C]

        Returns:
            tensor: Enhanced feature map [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Apply layer normalization
        x_norm = self.norm(x)

        # Reshape to 2D spatial format
        x_reshaped = x_norm.view(B, H, W, C)

        # Apply BSAN
        x_enhanced = self.bsan(x_reshaped)

        # Reshape back to sequence format
        x_enhanced = x_enhanced.view(B, H * W, C)

        # Residual connection
        x = x + x_enhanced

        return x