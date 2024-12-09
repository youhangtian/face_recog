import torch 
import torch.nn as nn 
from functools import partial 

from .vision_transformer import VisionTransformer 

def get_backbone(cfg):
    if cfg.network == 'vit_t':
        model = VisionTransformer(
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            freeze_patch_embed=cfg.freeze_patch_embed,
            num_features=cfg.num_features,
            embed_dim=192,
            depth=12,
            num_heads=3,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif cfg.network == 'vit_s':
        model = VisionTransformer(
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            freeze_patch_embed=cfg.freeze_patch_embed,
            num_features=cfg.num_features,
            embed_dim=384,
            depth=12,
            num_heads=6,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif cfg.network == 'vit_m':
        model = VisionTransformer(
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            freeze_patch_embed=cfg.freeze_patch_embed,
            num_features=cfg.num_features,
            embed_dim=512,
            depth=12,
            num_heads=8,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    elif cfg.network == 'vit_b':
        model = VisionTransformer(
            input_size=cfg.input_size,
            patch_size=cfg.patch_size,
            freeze_patch_embed=cfg.freeze_patch_embed,
            num_features=cfg.num_features,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

    else:
        raise ValueError(f'do not support backbone {cfg.network}')

    if cfg.ckpt:
        checkpoint = torch.load(cfg.ckpt)
        model.load_state_dict(checkpoint)

    return model
