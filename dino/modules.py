"""Modules used in the DINO model.

https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
"""
import functools
import math
from typing import Sequence
import warnings
import beartype
import torch
from torch import nn
from torchtyping import TensorType

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


def _compute_num_patches(img_size: int, patch_size: int) -> int:
    return (img_size // patch_size) * (img_size // patch_size)


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()
        num_patches = _compute_num_patches(img_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: TensorType["batch", "channel", "height", "width"]):
        # https://pytorch.org/vision/0.8/transforms.html
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def prepare_tokens(
    x: TensorType["batch", "channel", "width", "height"],
    patch_embed: PatchEmbed,
    cls_token: TensorType[1, 1, "embed_dim"],
    pos_embed: TensorType[1, "patches_1", "embed_dim"],
    pos_drop: nn.Dropout,
) -> torch.Tensor:
    B, nc, w, h = x.shape
    x = patch_embed(x)  # patch linear embedding

    # Add the [CLS] token to the embed patch tokens
    cls_tokens = cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # add positional encoding to each token
    x = x + interpolate_pos_encoding(
        pos_embed,
        n_patch=x.size(1) - 1,
        dim=x.size(-1),
        patch_size=patch_embed.patch_size,
        width=w,
        height=h,
    )

    return pos_drop(x)


def interpolate_pos_encoding(
    pos_embed: TensorType[1, "patch_1", "token"],
    n_patch: int,  # TODO: Is this dynamic?
    dim: int,  # TODO: Is this dynamic?
    patch_size: int,
    width: int,  # x.size(-2)
    height: int,  # x.size(-1)
):
    # B, nc, w, h = x.shape
    # n_patch = x.size(1) - 1
    # dim = x.size(-1)

    token_size = pos_embed.size(1) - 1  # The number of tokens (minus the cls token)
    if n_patch == token_size and width == height:
        # No need to interpolate
        return pos_embed

    class_pos_embed = pos_embed[:, 0]  # (dim, 1)
    patch_pos_embed = pos_embed[:, 1:]  # (dim, token_size)
    pos_embed_width = width // patch_size
    pos_embed_hight = height // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    # What is w0 and h0?
    # The position embedding is resized to a square
    pos_embed_side_length = int(math.sqrt(token_size))
    reshaped_patch_pos_embed = patch_pos_embed.reshape(
        1, pos_embed_side_length, pos_embed_side_length, dim
    )
    patch_pos_embed = nn.functional.interpolate(
        reshaped_patch_pos_embed.permute(
            0, 3, 1, 2
        ),  # (1, dim, pos_embed_side_length, pos_embed_side_length)
        size=(pos_embed_width, pos_embed_hight),
        mode="bicubic",
    )
    assert int(pos_embed_width) == patch_pos_embed.size(-2)
    assert int(pos_embed_hight) == patch_pos_embed.size(-1)
    patch_pos_embed = patch_pos_embed.permute(
        0, 2, 3, 1
    )  # (1, pos_embed_side_length, pos_embed_side_length, dim)
    patch_pos_embed = patch_pos_embed.view(
        1, -1, dim
    )  # (1, pos_embed_side_length*pos_embed_side_length, dim)
    class_pos_embed = class_pos_embed.unsqueeze(0)  # (1, 1, dim)
    # Put the class pos embedding at the first position
    return torch.cat((class_pos_embed, patch_pos_embed), dim=1)


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
    img_size: Sequence[int] = (224,),
    patch_size: int = 16,
    input_channels: int = 3,
    num_classes: int = 0,
    embed_dim: int = 768,
    depth: int = 12,
    num_heads: int = 12,
    mlp_ratio: float = 4.0,
    qkv_bias: bool = False,
    qk_scale=None,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    drop_path_rate: float = 0.0,
    norm_layer=nn.LayerNorm,
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