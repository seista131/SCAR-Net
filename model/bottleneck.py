import torch
import torch.nn as nn
from models.swin_transformer import SwinTransformerBlock


class ScarNetBottleneck(nn.Module):
    """
    Bottleneck module for SCAR-Net that connects the encoder and decoder
    It consists of a series of Swin Transformer blocks at the lowest resolution
    """

    def __init__(
            self,
            dim,
            input_resolution,
            depth=2,
            num_heads=24,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False
    ):
        super(ScarNetBottleneck, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # Build bottleneck blocks using Swin Transformer blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)
        ])

        # Final normalization layer
        self.norm = norm_layer(dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for layers"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        Forward pass of the bottleneck

        Args:
            x (tensor): Feature map from encoder [B, L, C]

        Returns:
            tensor: Processed features for decoder [B, L, C]
        """
        # Process through bottleneck blocks
        for blk in self.blocks:
            x = blk(x)

        # Apply final normalization
        x = self.norm(x)

        return x


class EnhancedBottleneck(nn.Module):
    """
    Enhanced bottleneck with additional feature processing capabilities
    """

    def __init__(
            self,
            dim,
            input_resolution,
            depth=2,
            num_heads=24,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False
    ):
        super(EnhancedBottleneck, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution

        # Main bottleneck blocks
        self.bottleneck = ScarNetBottleneck(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # Additional feature enhancement components
        self.channel_attention = ChannelAttention(dim)
        self.spatial_attention = SpatialAttention()

        # Final feature fusion
        self.fusion_norm = norm_layer(dim)

    def forward(self, x):
        """
        Forward pass of the enhanced bottleneck

        Args:
            x (tensor): Feature map from encoder [B, L, C]

        Returns:
            tensor: Enhanced features for decoder [B, L, C]
        """
        # Original bottleneck processing
        bottleneck_features = self.bottleneck(x)

        # Reshape for attention mechanisms (from NHWC to NCHW)
        B, L, C = bottleneck_features.shape
        H, W = self.input_resolution
        x_reshaped = bottleneck_features.permute(0, 2, 1).reshape(B, C, H, W)

        # Apply channel and spatial attention
        x_channel = self.channel_attention(x_reshaped)
        x_spatial = self.spatial_attention(x_channel)

        # Reshape back to NHWC format
        x_enhanced = x_spatial.reshape(B, C, H * W).permute(0, 2, 1)

        # Residual connection
        x_out = bottleneck_features + x_enhanced

        # Final normalization
        x_out = self.fusion_norm(x_out)

        return x_out


class ChannelAttention(nn.Module):
    """
    Channel attention module
    """

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial attention module
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)
