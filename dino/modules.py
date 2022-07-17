"""Modules used in the DINO model.

https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
"""
import functools
import math
from typing import Optional, Type
import warnings
import beartype
import torch
from torch import nn
from torchtyping import TensorType
import einops

SQRT_2 = math.sqrt(2.0)


@beartype.beartype
def _norm_cdf(x: float) -> float:
    """Compute standard normal cumulative distribution function."""
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
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = _compute_num_patches(img_size, patch_size)
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: TensorType["batch", "channel", "height", "width"]):
        # https://pytorch.org/vision/0.8/transforms.html
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    """Drop path for the DINO model."""
    if drop_prob <= 0.0 or not training:
        return x

    keep_prob = 1 - drop_prob
    # work with tensors of different dim, not just 2D ConvNets
    # TODO: Comment the shape out and do broadcasting instead
    shape = (x.size(0),) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        activation: Type[nn.Module] = nn.GELU,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.linear_1 = nn.Linear(in_features, hidden_features)
        self.activation = activation()
        self.linear_2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear_2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attention_drop_prob=0.0,
        proj_drop=0.0,
    ):
        """Create a multi-head attention layer.

        Args:
            dim: The dimension of the input.
            num_heads: The number of attention heads.
            qkv_bias: Whether to use bias in the query, key, value layers.
            qk_scale: The scale of the query layer.
            attention_drop_prob: The probability of dropout in the attention layer.
            proj_drop: The probability of dropout in the projection layer.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.drop = nn.Dropout(attention_drop_prob)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: TensorType["batch", "N", "channel"]):
        B = x.size(0)
        N = x.size(1)
        C = x.size(2)

        # NOTE: Why reshaping?
        qkv = (
            self.qkv(x)
            .reshape(
                B,
                N,
                3,
                self.num_heads,
                torch.div(C, self.num_heads, rounding_mode="trunc"),
            )
            .permute(2, 0, 3, 1, 4)
        )
        query, key, value = qkv[0], qkv[1], qkv[2]

        attn = (query @ key.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.drop(attn)

        x = (attn @ value).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        mlp_drop_prob=0.0,
        attention_drop_prob=0.0,
        drop_path_prob=0.0,
        activation=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attention_drop_prob=attention_drop_prob,
            proj_drop=mlp_drop_prob,
        )
        self.drop_path = (
            DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()
        )
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            activation=activation,
            drop_prob=mlp_drop_prob,
        )

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def prepare_tokens(
    x: TensorType["batch", "channel", "height", "width"],
    patch_embed: PatchEmbed,
    cls_token: TensorType[1, 1, "embed_dim"],
    pos_embed: TensorType[1, "patches_1", "embed_dim"],
    pos_drop: nn.Dropout,
    patch_size: int,
) -> torch.Tensor:
    """Prepare tokens for the transformer.

    Args:
        x: Input image.
        patch_embed: Patch embedding.
        cls_token: CLS token.
        pos_embed: Position embedding.
        pos_drop: Position dropout.
        patch_size: Patch size.

    Returns:
        Prepared tokens.
    """
    batch, _, height, width = x.shape
    x = patch_embed(x)  # Patch linear embedding

    # Add the [CLS] token to the embed patch tokens
    batch = x.size(0)
    cls_tokens = cls_token.expand(batch, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)

    # Add positional encoding to each token
    x = x + interpolate_pos_encoding(
        pos_embed,
        n_patch=x.size(1) - 1,
        dim=x.size(-1),
        patch_size=patch_size,
        height=height,
        width=width,
    )

    return pos_drop(x)


def _to_bchw(
    x: TensorType["batch", "height", "width", "channel"]
) -> TensorType["batch", "channel", "height", "width"]:
    # return x.permute(0, 3, 1, 2)
    return einops.rearrange(x, "b h w c -> b c h w")


def _to_bhwc(
    x: TensorType["batch", "channel", "height", "width"]
) -> TensorType["batch", "height", "width", "channel"]:
    # return x.permute(0, 2, 3, 1)
    return einops.rearrange(x, "b c h w -> b h w c")


def interpolate_pos_encoding(
    pos_embed: TensorType[1, "patch_1", "token"],
    n_patch: int,
    dim: int,
    patch_size: int,
    height: int,  # x.size(-2)
    width: int,  # x.size(-1)
):
    token_size = pos_embed.size(1) - 1  # The number of tokens (minus the cls token)
    if n_patch == token_size and height == width:
        # No need to interpolate
        return pos_embed

    class_pos_embed = pos_embed[:, 0]  # (dim, 1)
    patch_pos_embed = pos_embed[:, 1:]  # (dim, token_size)
    pos_embed_hight = height // patch_size
    pos_embed_width = width // patch_size
    # we add a small number to avoid floating point error in the interpolation
    # see discussion at https://github.com/facebookresearch/dino/issues/8
    # What is w0 and h0?
    # The position embedding is resized to a square
    pos_embed_side_length = int(math.sqrt(token_size))
    reshaped_patch_pos_embed = patch_pos_embed.reshape(
        1, pos_embed_side_length, pos_embed_side_length, dim
    )
    patch_pos_embed = nn.functional.interpolate(
        _to_bchw(
            reshaped_patch_pos_embed
        ),  # (1, dim, pos_embed_side_length, pos_embed_side_length)
        size=(pos_embed_hight, pos_embed_width),
        mode="bicubic",
    )
    assert int(pos_embed_hight) == patch_pos_embed.size(-2)
    assert int(pos_embed_width) == patch_pos_embed.size(-1)
    patch_pos_embed = _to_bhwc(
        patch_pos_embed
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


class VisionTransformer(nn.Module):
    """Vision Transformer"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
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
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        depth_decay_rule = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    mlp_drop_prob=drop_rate,
                    attention_drop_prob=attn_drop_rate,
                    drop_path_prob=depth_decay_rule[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = (
            nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        )

        _truncated_normal(self.pos_embed, std=0.02)
        _truncated_normal(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            _truncated_normal(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(
        self, x: TensorType["batch", "channel", "height", "width"]
    ) -> torch.Tensor:
        x = prepare_tokens(
            x,
            self.patch_embed,
            self.cls_token,
            self.pos_embed,
            self.pos_drop,
            x.size(0),
        )
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]


def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_small(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_base(patch_size=16, **kwargs):
    return VisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
