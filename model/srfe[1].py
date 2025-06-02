import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """
    Basic convolutional block with batch normalization and ReLU activation
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


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


class ScarFeatureEnhancer(nn.Module):
    """
    Scar Feature Enhancer (SFE) module
    """

    def __init__(self, channels):
        super(ScarFeatureEnhancer, self).__init__()
        # Half the input channels for each branch
        self.channels = channels // 2

        # Three convolutional blocks for feature extraction
        self.block1 = ConvBlock(self.channels, self.channels)
        self.block2 = ConvBlock(self.channels, self.channels)
        self.block3 = ConvBlock(self.channels, self.channels)

        # Output projection with 1x1 convolution
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels)
        )

    def forward(self, x):
        # Store original input for residual connection
        identity = x

        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Apply output projection
        x = self.output_proj(x)

        # Add residual connection
        x = x + identity

        return x


class RecurrenceFeatureEnhancer(nn.Module):
    """
    Recurrence Feature Enhancer (RFE) module
    """

    def __init__(self, channels):
        super(RecurrenceFeatureEnhancer, self).__init__()
        # Half the input channels for each branch
        self.channels = channels // 2

        # Three convolutional blocks for feature extraction
        self.block1 = ConvBlock(self.channels, self.channels)
        self.block2 = ConvBlock(self.channels, self.channels)
        self.block3 = ConvBlock(self.channels, self.channels)

        # Output projection with 1x1 convolution
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, kernel_size=1),
            nn.BatchNorm2d(self.channels)
        )

    def forward(self, x):
        # Store original input for residual connection
        identity = x

        # Apply convolutional blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        # Apply output projection
        x = self.output_proj(x)

        # Add residual connection
        x = x + identity

        return x


class SRFE(nn.Module):
    """
    Scar-Recurrence Feature Enhancer (SRFE) module
    """

    def __init__(self, channels, stage=None):
        super(SRFE, self).__init__()
        self.channels = channels
        self.stage = stage  # Stage identifier (1, 2, or 3)

        # Feature fusion for combining decoder features with previous SRFE output
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        # SFE and RFE branches
        self.sfe = ScarFeatureEnhancer(channels)
        self.rfe = RecurrenceFeatureEnhancer(channels)

        # Channel-wise attention for feature integration
        self.se_block = SqueezeExcitation(channels)

    def forward(self, x, prev_srfe=None):
        """
        Forward pass of the SRFE module

        Args:
            x (tensor): Input feature map from decoder [B, C, H, W]
            prev_srfe (tensor, optional): Output from previous SRFE module, upsampled to current resolution

        Returns:
            tensor: Enhanced feature map [B, C, H, W]
        """
        # Reshape if input is in NHWC format (from Swin Transformer)
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)
            nhwc_format = True
        else:
            nhwc_format = False

        # If previous SRFE output is available, fuse it with current features
        if prev_srfe is not None:
            # Ensure prev_srfe has the same spatial dimensions as x
            if prev_srfe.shape[2:] != x.shape[2:]:
                prev_srfe = F.interpolate(prev_srfe, size=x.shape[2:], mode='bilinear', align_corners=False)

            # Concatenate along channel dimension and apply fusion
            x = torch.cat([x, prev_srfe], dim=1)
            x = self.feature_fusion(x)

        # Store fused features for residual connection
        identity = x

        # Split channels for SFE and RFE branches
        c = self.channels // 2
        x_sfe = x[:, :c, :, :]
        x_rfe = x[:, c:, :, :]

        # Apply SFE and RFE branches
        x_sfe = self.sfe(x_sfe)
        x_rfe = self.rfe(x_rfe)

        # Concatenate branch outputs
        x = torch.cat([x_sfe, x_rfe], dim=1)

        # Apply channel-wise attention
        x = self.se_block(x)

        # Add residual connection
        x = x + identity

        # Convert back to NHWC format if input was in that format
        if nhwc_format:
            B, C, H, W = x.shape
            x = x.reshape(B, C, H * W).permute(0, 2, 1)

        return x


class SRFELayer(nn.Module):
    """
    SRFE Layer adapted for Swin Transformer architecture
    Handles input in NHWC format (B, H*W, C) that Swin Transformer uses
    """

    def __init__(self, dim, input_resolution, stage=None):
        super(SRFELayer, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.stage = stage

        # Core SRFE module
        self.srfe = SRFE(dim, stage)

        # Normalization layer
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, prev_srfe=None):
        """
        Forward pass

        Args:
            x (tensor): Input feature map in NHWC format [B, H*W, C]
            prev_srfe (tensor, optional): Output from previous SRFE module

        Returns:
            tensor: Enhanced feature map [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        # Apply layer normalization
        x_norm = self.norm(x)

        # Reshape to 2D spatial format
        x_reshaped = x_norm.permute(0, 2, 1).reshape(B, C, H, W)

        if prev_srfe is not None and prev_srfe.dim() == 3:
            # Convert prev_srfe from NHWC to NCHW if needed
            prev_H, prev_W = int((prev_srfe.shape[1]) ** 0.5), int((prev_srfe.shape[1]) ** 0.5)
            prev_srfe = prev_srfe.permute(0, 2, 1).reshape(prev_srfe.shape[0], prev_srfe.shape[2], prev_H, prev_W)

        # Apply SRFE
        x_enhanced = self.srfe(x_reshaped, prev_srfe)

        # Convert back to NHWC format if SRFE didn't already do it
        if x_enhanced.dim() == 4:
            x_enhanced = x_enhanced.reshape(B, C, H * W).permute(0, 2, 1)

        # Residual connection
        x = x + x_enhanced

        return x