from typing import Dict

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY


@SEM_SEG_HEADS_REGISTRY.register()
class SingleFeaturePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        feature_name: str,
        mask_dim: int,
        conv_dim: int,
    ):
        super().__init__()
        
        self.feature_name = feature_name
        self.mask_dim = mask_dim
        self.conv_dim = conv_dim

        input_channels = input_shape[feature_name].channels

        self.mask_upscale_layer = nn.Sequential(
            nn.ConvTranspose2d(input_channels, mask_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, mask_dim),
        )
        weight_init.c2_xavier_fill(self.mask_upscale_layer[0])
        nn.init.constant_(self.mask_upscale_layer[1].weight, 1)
        nn.init.constant_(self.mask_upscale_layer[1].bias, 0)

        self.feature_upscale_layer = nn.Sequential(
            nn.ConvTranspose2d(input_channels, conv_dim, kernel_size=2, stride=2),
            nn.GroupNorm(32, conv_dim),
        )
        weight_init.c2_xavier_fill(self.feature_upscale_layer[0])
        nn.init.constant_(self.feature_upscale_layer[1].weight, 1)
        nn.init.constant_(self.feature_upscale_layer[1].bias, 0)


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        assert (
            len(cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES) == 1
        ), "SingleFeaturePixelDecoder only supports a single feature"
        
        ret = {}
        ret["input_shape"] = input_shape
        ret["feature_name"] = cfg.MODEL.SEM_SEG_HEAD.DEFORMABLE_TRANSFORMER_ENCODER_IN_FEATURES[0]
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        return ret

    @autocast(enabled=False)
    def forward_features(self, features):
        x = features[self.feature_name].float()

        # Upscale mask features
        up_mask_features = self.mask_upscale_layer(x)
        
        # Upscale output patch features
        up_features = self.feature_upscale_layer(x)
        
        return up_mask_features, up_features, [up_features]