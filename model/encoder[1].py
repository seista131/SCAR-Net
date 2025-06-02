import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import PatchEmbed, BasicSwinLayer, PatchMerging
from models.bsan import BSANForSwinTransformer


class ScarNetEncoder(nn.Module):
    """
    SCAR-Net Encoder module that incorporates Swin Transformer blocks and
    Boundary Sensitive Attention Networks (BSAN).
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            in_chans=3,
            embed_dim=96,
            depths=[2, 2, 6],
            num_heads=[3, 6, 12],
            window_size=7,
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm,
            patch_norm=True,
            use_checkpoint=False
    ):
        super(ScarNetEncoder, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
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
        self.patch_norm = patch_norm
        self.use_checkpoint = use_checkpoint
        self.num_layers = len(depths)

        # Patch partition and embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if patch_norm else None
        )

        # Calculate patch resolution
        self.patches_resolution = self.patch_embed.patches_resolution
        num_patches = self.patch_embed.num_patches

        # Absolute position embedding is not used in Swin Transformer
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build encoder stages
        self.layers = nn.ModuleList()
        self.bsan_layers = nn.ModuleList()

        # Current resolution for each stage
        self.resolutions = []

        # Current dimension for each stage
        self.dims = []

        for i_layer in range(self.num_layers):
            # Calculate current resolution
            layer_resolution = [
                self.patches_resolution[0] // (2 ** i_layer),
                self.patches_resolution[1] // (2 ** i_layer)
            ]
            self.resolutions.append(layer_resolution)

            # Calculate current dimension
            layer_dim = embed_dim * (2 ** i_layer)
            self.dims.append(layer_dim)

            # Create Swin Transformer layer
            layer = BasicSwinLayer(
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
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint
            )
            self.layers.append(layer)

            # Create BSAN layer for this stage
            bsan = BSANForSwinTransformer(
                dim=layer_dim,
                input_resolution=layer_resolution
            )
            self.bsan_layers.append(bsan)

        # Final normalization layer
        self.norm = norm_layer(self.dims[-1])

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
        Forward pass of the SCAR-Net encoder

        Args:
            x (tensor): Input image [B, C, H, W]

        Returns:
            list: List of feature maps from each encoder stage after BSAN enhancement
            tensor: Final bottleneck features
        """
        # Store intermediate features for skip connections
        features = []

        # Patch embedding
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        # Process through encoder stages
        for i, (layer, bsan) in enumerate(zip(self.layers, self.bsan_layers)):
            # Apply Swin Transformer layer
            x = layer(x)

            # Apply BSAN
            enhanced_x = bsan(x)

            # Store enhanced features for skip connections
            features.append(enhanced_x)

        # Apply final normalization
        x = self.norm(x)

        return features, x


class ScarNetEncoderStage(nn.Module):
    """
    A single stage of the SCAR-Net encoder, combining Swin Transformer blocks and BSAN
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
            downsample=None,
            use_checkpoint=False
    ):
        super(ScarNetEncoderStage, self).__init__()

        # Swin Transformer layer
        self.swin_layer = BasicSwinLayer(
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
            downsample=downsample,
            use_checkpoint=use_checkpoint
        )

        # Boundary Sensitive Attention Network
        self.bsan = BSANForSwinTransformer(
            dim=dim,
            input_resolution=input_resolution
        )

    def forward(self, x):
        """
        Forward pass of a single encoder stage

        Args:
            x (tensor): Input features [B, L, C]

        Returns:
            tuple: (enhanced_features, downsampled_features)
                enhanced_features: Features after BSAN for skip connection
                downsampled_features: Downsampled features for next stage
        """
        # Apply Swin Transformer layer
        x_downsampled = self.swin_layer(x)

        # If downsampling occurred, we need to apply BSAN before downsampling
        if self.swin_layer.downsample is not None:
            # Store the resolution before downsampling
            H, W = self.swin_layer.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            # Apply BSAN to features before downsampling
            enhanced_x = self.bsan(x)

            return enhanced_x, x_downsampled
        else:
            # Apply BSAN to final features (no downsampling)
            enhanced_x = self.bsan(x_downsampled)

            return enhanced_x, enhanced_x