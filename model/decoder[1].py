import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import SwinTransformerBlock
from models.srfe import SRFELayer


class PatchExpanding(nn.Module):
    """
    Patch Expanding Layer for upsampling in decoder
    """

    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, dim_scale * dim, bias=False)
        self.norm = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, f"input feature has wrong size {L} vs {H * W}"

        x = self.expand(x)  # B, H*W, C*dim_scale

        # Reshape to spatial dimensions
        x = x.view(B, H, W, C * self.dim_scale)

        # Rearrange for upsampling
        x = x.view(B, H, W, self.dim_scale, self.dim_scale, C // self.dim_scale)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H * self.dim_scale, W * self.dim_scale, C // self.dim_scale)

        # Flatten to sequence format
        x = x.view(B, -1, C // self.dim_scale)

        # Apply normalization
        x = self.norm(x)

        return x


class FinalUpsampling(nn.Module):
    """
    Final upsampling layer to restore original image resolution
    """

    def __init__(self, in_channels, out_channels, scale_factor=4):
        super(FinalUpsampling, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: Features in format [B, H*W, C] or [B, C, H, W]
        """
        # Convert to NCHW format if needed
        if x.dim() == 3:
            B, L, C = x.shape
            H = W = int(L ** 0.5)
            x = x.permute(0, 2, 1).reshape(B, C, H, W)

        # Apply 1x1 convolution
        x = self.conv(x)

        # Upsample to original resolution
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)

        return x


class DecoderBlock(nn.Module):
    """
    Decoder block with Swin Transformer and feature fusion
    """

    def __init__(
            self,
            dim,
            input_resolution,
            depth,
            num_heads,
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop=0.,
            attn_drop=0.,
            drop_path=0.,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
            stage=None  # For SRFE stage identification
    ):
        super(DecoderBlock, self).__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.stage = stage

        # Swin Transformer blocks
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

        # Upsampling layer
        if upsample is not None:
            self.upsample = upsample(
                input_resolution=input_resolution,
                dim=dim
            )
        else:
            self.upsample = None

        # SRFE layer
        self.srfe = SRFELayer(
            dim=dim,
            input_resolution=input_resolution,
            stage=stage
        )

    def forward(self, x, skip_features=None, prev_srfe_output=None):
        """
        Forward pass of decoder block

        Args:
            x (tensor): Input features from previous decoder stage [B, L, C]
            skip_features (tensor, optional): Skip connection features from encoder
            prev_srfe_output (tensor, optional): Output from previous SRFE module

        Returns:
            tuple: (x, srfe_output)
                x: Output features from this decoder stage
                srfe_output: Output from SRFE module for next stage
        """
        # Apply Swin Transformer blocks
        for blk in self.blocks:
            x = blk(x)

        # Fuse with skip connection if available
        if skip_features is not None:
            x = x + skip_features

        # Store features before upsampling for SRFE
        features_for_srfe = x

        # Apply upsampling if available
        if self.upsample is not None:
            x = self.upsample(x)

        # Apply SRFE
        srfe_output = self.srfe(features_for_srfe, prev_srfe_output)

        return x, srfe_output


class ScarNetDecoder(nn.Module):
    """
    SCAR-Net Decoder module incorporating Swin Transformer blocks and SRFE modules
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            embed_dim=96,
            depths=[6, 2, 2],
            num_heads=[12, 6, 3],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            use_checkpoint=False,
            num_classes=1
    ):
        super(ScarNetDecoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.norm_layer = norm_layer
        self.num_layers = len(depths)
        self.num_classes = num_classes

        # Calculate patch resolution
        self.patches_resolution = [img_size // patch_size, img_size // patch_size]

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build decoder stages
        self.layers = nn.ModuleList()

        # Resolutions and dimensions for each stage
        self.resolutions = []
        self.dims = []

        for i_layer in range(self.num_layers):
            # Calculate current resolution (going from smaller to larger)
            layer_resolution = [
                self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))
            ]
            self.resolutions.append(layer_resolution)

            # Calculate current dimension (going from larger to smaller)
            layer_dim = embed_dim * (2 ** (self.num_layers - 1 - i_layer))
            self.dims.append(layer_dim)

            # Create decoder block
            layer = DecoderBlock(
                dim=layer_dim,
                input_resolution=layer_resolution,
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                upsample=PatchExpanding if i_layer < self.num_layers - 1 else None,
                use_checkpoint=use_checkpoint,
                stage=self.num_layers - i_layer  # Stage 3, 2, 1 for SRFE
            )
            self.layers.append(layer)

        # Final upsampling to original image resolution
        self.final_upsample = FinalUpsampling(
            in_channels=embed_dim,
            out_channels=num_classes,
            scale_factor=patch_size
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

    def forward(self, x, encoder_features):
        """
        Forward pass of the SCAR-Net decoder

        Args:
            x (tensor): Bottleneck features [B, L, C]
            encoder_features (list): List of feature maps from encoder stages

        Returns:
            tuple: (segmentation, srfe_outputs)
                segmentation: Final segmentation map [B, num_classes, H, W]
                srfe_outputs: List of outputs from SRFE modules
        """
        # Store SRFE outputs for later use
        srfe_outputs = []
        prev_srfe = None

        # Process through decoder stages
        for i, layer in enumerate(self.layers):
            # Get skip connection from encoder (in reverse order)
            skip = encoder_features[self.num_layers - 1 - i] if i < len(encoder_features) else None

            # Apply decoder block
            x, srfe_output = layer(x, skip, prev_srfe)

            # Update previous SRFE output
            prev_srfe = srfe_output

            # Store SRFE output
            srfe_outputs.append(srfe_output)

        # Apply final upsampling for segmentation
        segmentation = self.final_upsample(x)

        return segmentation, srfe_outputs