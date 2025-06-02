import torch
import torch.nn as nn
import torch.nn.functional as F
from models.swin_transformer import PatchEmbed, SwinTransformerBlock, window_partition, window_reverse


class MFTL(nn.Module):
    """
    Multi-feature Transformer Layer (MFTL) Architecture for fusing backbone and SRFE features
    and generating binary classification outputs for scar presence and recurrence.
    """

    def __init__(
            self,
            img_size=224,
            patch_size=4,
            backbone_dim=96,
            srfe_dim=96,
            fusion_dim=128,
            window_size=7,
            depths=[2, 2],
            num_heads=[8, 16],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.1,
            norm_layer=nn.LayerNorm
    ):
        super(MFTL, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone_dim = backbone_dim
        self.srfe_dim = srfe_dim
        self.fusion_dim = fusion_dim
        self.window_size = window_size
        self.depths = depths
        self.num_heads = num_heads

        # Feature projection to map concatenated features to fusion dimension
        self.feature_projection = nn.Conv2d(backbone_dim + srfe_dim, fusion_dim, kernel_size=1)

        # Patch partitioning and embedding
        self.patch_partition = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=fusion_dim,
            embed_dim=fusion_dim,
            norm_layer=norm_layer
        )

        # Calculate reduced resolution
        self.patches_resolution = self.patch_partition.patches_resolution

        # Set up drop path rate for different layers
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # First Swin Transformer block stage
        self.stage1_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=fusion_dim,
                input_resolution=self.patches_resolution,
                num_heads=num_heads[0],
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer
            ) for i in range(depths[0])
        ])

        # Patch merging layer
        self.patch_merging = nn.Sequential(
            norm_layer(fusion_dim),
            nn.Linear(fusion_dim, 2 * fusion_dim, bias=False)
        )

        # Smaller resolution after patch merging
        self.merged_resolution = [res // 2 for res in self.patches_resolution]

        # Second Swin Transformer block stage
        self.stage2_blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=2 * fusion_dim,
                input_resolution=self.merged_resolution,
                num_heads=num_heads[1],
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[depths[0] + i],
                norm_layer=norm_layer
            ) for i in range(depths[1])
        ])

        # Global pooling and normalization
        self.norm = norm_layer(2 * fusion_dim)

        # Classification heads
        self.scar_classifier = nn.Linear(2 * fusion_dim, 1)
        self.recurrence_classifier = nn.Linear(2 * fusion_dim, 1)

        # Sigmoid activation for binary outputs
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear and conv layers"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _patch_merging(self, x, H, W):
        """Custom patch merging function to handle sequence data"""
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Gather patches
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        # Concatenate along feature dimension
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # Apply normalization and projection
        x = self.patch_merging[0](x)
        x = self.patch_merging[1](x)

        return x

    def forward(self, backbone_features, srfe_features):
        """
        Forward pass of the MFTL

        Args:
            backbone_features (tensor): Features from backbone network [B, H*W, C]
            srfe_features (tensor): Features from SRFE modules [B, H*W, C]

        Returns:
            tuple: (scar_pred, recurrence_pred) - Binary prediction scores for scar presence and recurrence
        """
        # Convert from NHWC to NCHW if needed
        if backbone_features.dim() == 3:
            B, L, C = backbone_features.shape
            H = W = int(L ** 0.5)
            backbone_features = backbone_features.permute(0, 2, 1).reshape(B, C, H, W)

        if srfe_features.dim() == 3:
            B, L, C = srfe_features.shape
            H = W = int(L ** 0.5)
            srfe_features = srfe_features.permute(0, 2, 1).reshape(B, C, H, W)

        # Ensure both features have the same spatial dimensions
        if backbone_features.shape[2:] != srfe_features.shape[2:]:
            srfe_features = F.interpolate(srfe_features, size=backbone_features.shape[2:],
                                          mode='bilinear', align_corners=False)

        # Concatenate features along channel dimension
        combined_features = torch.cat([backbone_features, srfe_features], dim=1)

        # Project to fusion dimension
        fused_features = self.feature_projection(combined_features)

        # Patch partitioning and embedding
        x = self.patch_partition(fused_features)

        # Get resolution after patch embedding
        Wh, Ww = self.patches_resolution

        # First stage Swin Transformer blocks
        for blk in self.stage1_blocks:
            x = blk(x)

        # Patch merging
        x = self._patch_merging(x, Wh, Ww)

        # Get resolution after patch merging
        Wh, Ww = self.merged_resolution

        # Second stage Swin Transformer blocks
        for blk in self.stage2_blocks:
            x = blk(x)

        # Global pooling (mean across tokens)
        x = x.mean(dim=1)

        # Apply normalization
        x = self.norm(x)

        # Classification heads
        scar_pred = self.sigmoid(self.scar_classifier(x))
        recurrence_pred = self.sigmoid(self.recurrence_classifier(x))

        return scar_pred, recurrence_pred


class SimplifiedMFTL(nn.Module):
    """
    A simplified version of MFTL that uses a more straightforward approach
    for feature fusion and classification
    """

    def __init__(
            self,
            backbone_dim=96,
            srfe_dim=96,
            fusion_dim=128,
            mlp_ratio=4.,
            drop_rate=0.1,
            norm_layer=nn.LayerNorm
    ):
        super(SimplifiedMFTL, self).__init__()

        self.backbone_dim = backbone_dim
        self.srfe_dim = srfe_dim
        self.fusion_dim = fusion_dim

        # Feature projection for backbone and SRFE features
        self.backbone_projection = nn.Conv2d(backbone_dim, fusion_dim // 2, kernel_size=1)
        self.srfe_projection = nn.Conv2d(srfe_dim, fusion_dim // 2, kernel_size=1)

        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Feature normalization
        self.norm = norm_layer(fusion_dim)

        # MLP head for classification
        hidden_dim = int(fusion_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(drop_rate)
        )

        # Classification heads
        self.scar_classifier = nn.Linear(hidden_dim // 2, 1)
        self.recurrence_classifier = nn.Linear(hidden_dim // 2, 1)

        # Sigmoid activation for binary outputs
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights for linear and conv layers"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, backbone_features, srfe_features):
        """
        Forward pass of the simplified MFTL

        Args:
            backbone_features (tensor): Features from backbone network [B, H*W, C] or [B, C, H, W]
            srfe_features (tensor): Features from SRFE modules [B, H*W, C] or [B, C, H, W]

        Returns:
            tuple: (scar_pred, recurrence_pred) - Binary prediction scores for scar presence and recurrence
        """
        # Convert from NHWC to NCHW if needed
        if backbone_features.dim() == 3:
            B, L, C = backbone_features.shape
            H = W = int(L ** 0.5)
            backbone_features = backbone_features.permute(0, 2, 1).reshape(B, C, H, W)

        if srfe_features.dim() == 3:
            B, L, C = srfe_features.shape
            H = W = int(L ** 0.5)
            srfe_features = srfe_features.permute(0, 2, 1).reshape(B, C, H, W)

        # Ensure both features have the same spatial dimensions
        if backbone_features.shape[2:] != srfe_features.shape[2:]:
            srfe_features = F.interpolate(srfe_features, size=backbone_features.shape[2:],
                                          mode='bilinear', align_corners=False)

        # Project features to lower dimension
        backbone_features = self.backbone_projection(backbone_features)
        srfe_features = self.srfe_projection(srfe_features)

        # Concatenate features along channel dimension
        fused_features = torch.cat([backbone_features, srfe_features], dim=1)

        # Global pooling
        pooled_features = self.global_avg_pool(fused_features).flatten(1)

        # Apply normalization
        normalized_features = self.norm(pooled_features)

        # Apply MLP
        features = self.mlp(normalized_features)

        # Classification heads
        scar_pred = self.sigmoid(self.scar_classifier(features))
        recurrence_pred = self.sigmoid(self.recurrence_classifier(features))

        return scar_pred, recurrence_pred