"""Modules used in the DINO model.

https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
"""
import functools
import math
from typing import Sequence, Type
import warnings
import beartype
import torch
from torch import nn

SQRT_2 = math.sqrt(2.0)


@beartype.beartype
def _norm_cdf(x: float) -> float:
    """Computes standard normal cumulative distribution function."""
    return (1.0 + math.erf(x / SQRT_2)) / 2.0


@beartype.beartype
def _truncated_normal(
    tensor, mean: float = 0, std: float = 1, a: float = -2, b: float = 2
) -> torch.Tensor:
    """Truncated normal distribution.

    Args:
        tensor: Tensor to hold samples from the distribution.
        mean: Mean of the distribution.
        std: Standard deviation of the distribution.
        a:
        b:

    https://github.com/facebookresearch/dino/blob/main/utils.py
    Method based on https://people.sc.fsu.edu/~jburkardt/presentations/_truncated_normal.pdf
    """
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        lower = _norm_cdf((a - mean) / std)
        upper = _norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * lower - 1, 2 * upper - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


@beartype.beartype
def _init_dino_head(linear_module: nn.Linear):
    _truncated_normal(linear_module.weight, std=0.02)
    if linear_module.bias is not None:
        nn.init.constant_(linear_module.bias, 0)


class Normalize(nn.Module):
    def __init__(self, p=2.0, dim=1, eps=1e-12, out=None):
        super().__init__()
        self.p = p
        self.dim = dim
        self.eps = eps
        self.out = out

    def forward(self, x):
        return torch.nn.functional.normalize(
            x, p=self.p, dim=self.dim, eps=self.eps, out=self.out
        )


@beartype.beartype
def dino_head(
    input_dim: int,
    output_dim: int,
    n_layers: int = 3,
    hidden_dim: int = 2048,
    bottleneck_dim: int = 256,
) -> nn.Module:
    """Creates the DINO head."""
    layers = []
    if n_layers == 1:
        layers.append(nn.Linear(input_dim, bottleneck_dim))
    else:
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))
    module = nn.ModuleList(*layers)

    # Initialize the weights
    for layer in module:
        if isinstance(layer, nn.Linear):
            _init_dino_head(layer)

    last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, output_dim, bias=False))
    last_layer.weight_g.data.fill_(1)
    last_layer.weight_g.requires_grad = False

    # TODO: Give the module a name
    return nn.Sequential(module, Normalize(dim=-1, p=2), last_layer)


def vision_transformer(
    img_size: Sequence[int]=(224,),
    patch_size: int=16,
    input_channels: int=3,
    num_classes: int=0,
    embed_dim: int=768,
    depth: int=12,
    num_heads: int=12,
    mlp_ratio: float=4.0,
    qkv_bias: bool=False,
    qk_scale=None,
    drop_rate: float=0.0,
    attn_drop_rate: float=0.0,
    drop_path_rate: float=0.0,
    norm_layer: Type[nn.LayerNorm]=nn.LayerNorm,
    **kwargs
):
    pass


def vit_tiny(patch_size=16, **kwargs):
    return vision_transformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def vit_small(patch_size=16, **kwargs):
    return vision_transformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )


def vit_base(patch_size=16, **kwargs):
    return vision_transformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs
    )
