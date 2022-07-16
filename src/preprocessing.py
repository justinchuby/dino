"""Data augmentation and preprocessing functions."""

import torch
from torch import nn
import math
from torchtyping import TensorType


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
