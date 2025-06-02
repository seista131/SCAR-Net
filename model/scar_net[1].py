import torch
import torch.nn as nn
import torch.nn.functional as F
from models.encoder import ScarNetEncoder
from models.decoder import ScarNetDecoder
from models.bottleneck import ScarNetBottleneck
from models.mftl import MFTL, SimplifiedMFTL


class SCARNet(nn.Module):
    """
    SCAR-Net: A Swin Transformer-based architecture for myocardial scar segmentation
    and recurrence prediction in cardiac MRI.
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=96,
            encoder_depths=[2, 2, 6],
            encoder_num_heads=[3, 6, 12],
            decoder_depths=[6, 2, 2],
            decoder_num_heads=[12, 6, 3],
            bottleneck_depth=2,
            bottleneck_num_heads=24,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            use_simplified_mftl=False,
            fusion_dim=128
    ):
        super(SCARNet, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size

        # Encoder
        self.encoder = ScarNetEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint
        )

        # Calculate bottleneck dimensions
        bottleneck_dim = embed_dim * (2 ** (len(encoder_depths) - 1))
        bottleneck_resolution = [
            img_size // patch_size // (2 ** (len(encoder_depths) - 1)),
            img_size // patch_size // (2 ** (len(encoder_depths) - 1))
        ]

        # Bottleneck
        self.bottleneck = ScarNetBottleneck(
            dim=bottleneck_dim,
            input_resolution=bottleneck_resolution,
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # Decoder
        self.decoder = ScarNetDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=decoder_depths,
            num_heads=decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            num_classes=num_classes
        )

        # Multi-feature Transformer Layer (MFTL)
        if use_simplified_mftl:
            self.mftl = SimplifiedMFTL(
                backbone_dim=embed_dim,
                srfe_dim=embed_dim,
                fusion_dim=fusion_dim,
                mlp_ratio=mlp_ratio,
                drop_rate=drop_rate,
                norm_layer=norm_layer
            )
        else:
            self.mftl = MFTL(
                img_size=img_size,
                patch_size=patch_size,
                backbone_dim=embed_dim,
                srfe_dim=embed_dim,
                fusion_dim=fusion_dim,
                window_size=window_size,
                depths=[2, 2],
                num_heads=[8, 16],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate,
                norm_layer=norm_layer
            )

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

    def upsample_and_convert(self, x, target_size=None):
        """
        Upsample feature maps to target size and convert from NHWC to NCHW format if needed
        """
        # Convert to NCHW format if needed
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # Upsample if target size is provided
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        """
        Forward pass of SCAR-Net

        Args:
            x (tensor): Input image [B, C, H, W]

        Returns:
            dict: Dictionary containing:
                'segmentation': Scar segmentation map [B, num_classes, H, W]
                'scar_prob': Probability of scar presence [B, 1]
                'recurrence_prob': Probability of recurrence [B, 1]
        """
        # Store input size for upsampling
        input_size = (x.shape[2], x.shape[3])

        # Encoder
        encoder_features, bottleneck_features = self.encoder(x)

        # Bottleneck
        bottleneck_output = self.bottleneck(bottleneck_features)

        # Decoder
        segmentation, srfe_outputs = self.decoder(bottleneck_output, encoder_features)

        # Get final features for classification
        final_backbone_features = encoder_features[0]  # First stage encoder features
        final_srfe_features = srfe_outputs[-1]  # Last SRFE module output

        # Convert and upsample to patch size if needed
        patch_resolution = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        backbone_features_processed = self.upsample_and_convert(final_backbone_features, patch_resolution)
        srfe_features_processed = self.upsample_and_convert(final_srfe_features, patch_resolution)

        # Apply MFTL for classification
        scar_prob, recurrence_prob = self.mftl(backbone_features_processed, srfe_features_processed)

        # Ensure segmentation is upsampled to input size
        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(segmentation, size=input_size, mode='bilinear', align_corners=False)

        # Return outputs as dictionary
        return {
            'segmentation': segmentation,
            'scar_prob': scar_prob,
            'recurrence_prob': recurrence_prob
        }


class SCARNetLite(nn.Module):
    """
    Lighter version of SCAR-Net with reduced complexity
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            num_classes=1,
            embed_dim=64,  # Reduced from 96
            encoder_depths=[2, 2, 4],  # Reduced depths
            encoder_num_heads=[2, 4, 8],  # Reduced heads
            decoder_depths=[4, 2, 2],
            decoder_num_heads=[8, 4, 2],
            bottleneck_depth=2,
            bottleneck_num_heads=16,  # Reduced from 24
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False,
            use_simplified_mftl=True,  # Default to simplified MFTL
            fusion_dim=96  # Reduced from 128
    ):
        super(SCARNetLite, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.window_size = window_size

        # Encoder
        self.encoder = ScarNetEncoder(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=encoder_depths,
            num_heads=encoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            patch_norm=patch_norm,
            use_checkpoint=use_checkpoint
        )

        # Calculate bottleneck dimensions
        bottleneck_dim = embed_dim * (2 ** (len(encoder_depths) - 1))
        bottleneck_resolution = [
            img_size // patch_size // (2 ** (len(encoder_depths) - 1)),
            img_size // patch_size // (2 ** (len(encoder_depths) - 1))
        ]

        # Bottleneck
        self.bottleneck = ScarNetBottleneck(
            dim=bottleneck_dim,
            input_resolution=bottleneck_resolution,
            depth=bottleneck_depth,
            num_heads=bottleneck_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint
        )

        # Decoder
        self.decoder = ScarNetDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=embed_dim,
            depths=decoder_depths,
            num_heads=decoder_num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
            num_classes=num_classes
        )

        # Use simplified MFTL for classification
        self.mftl = SimplifiedMFTL(
            backbone_dim=embed_dim,
            srfe_dim=embed_dim,
            fusion_dim=fusion_dim,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            norm_layer=norm_layer
        )

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

    def upsample_and_convert(self, x, target_size=None):
        """
        Upsample feature maps to target size and convert from NHWC to NCHW format if needed
        """
        # Convert to NCHW format if needed
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # Upsample if target size is provided
        if target_size is not None:
            x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)

        return x

    def forward(self, x):
        """
        Forward pass of SCARNetLite

        Args:
            x (tensor): Input image [B, C, H, W]

        Returns:
            dict: Dictionary containing:
                'segmentation': Scar segmentation map [B, num_classes, H, W]
                'scar_prob': Probability of scar presence [B, 1]
                'recurrence_prob': Probability of recurrence [B, 1]
        """
        # Store input size for upsampling
        input_size = (x.shape[2], x.shape[3])

        # Encoder
        encoder_features, bottleneck_features = self.encoder(x)

        # Bottleneck
        bottleneck_output = self.bottleneck(bottleneck_features)

        # Decoder
        segmentation, srfe_outputs = self.decoder(bottleneck_output, encoder_features)

        # Get final features for classification
        final_backbone_features = encoder_features[0]  # First stage encoder features
        final_srfe_features = srfe_outputs[-1]  # Last SRFE module output

        # Convert and upsample to patch size if needed
        patch_resolution = (self.img_size // self.patch_size, self.img_size // self.patch_size)
        backbone_features_processed = self.upsample_and_convert(final_backbone_features, patch_resolution)
        srfe_features_processed = self.upsample_and_convert(final_srfe_features, patch_resolution)

        # Apply MFTL for classification
        scar_prob, recurrence_prob = self.mftl(backbone_features_processed, srfe_features_processed)

        # Ensure segmentation is upsampled to input size
        if segmentation.shape[2:] != input_size:
            segmentation = F.interpolate(segmentation, size=input_size, mode='bilinear', align_corners=False)

        # Return outputs as dictionary
        return {
            'segmentation': segmentation,
            'scar_prob': scar_prob,
            'recurrence_prob': recurrence_prob
        }


def create_scar_net(model_type='standard', **kwargs):
    """
    Factory function to create different variants of SCAR-Net

    Args:
        model_type (str): Model type - 'standard', 'lite', or 'custom'
        **kwargs: Additional arguments to override default configuration

    Returns:
        nn.Module: SCAR-Net model instance
    """
    if model_type == 'lite':
        return SCARNetLite(**kwargs)
    elif model_type == 'custom':
        return SCARNet(**kwargs)
    else:  # 'standard'
        return SCARNet(**kwargs)