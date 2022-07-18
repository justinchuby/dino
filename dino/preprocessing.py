"""Data augmentation and preprocessing functions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torchtyping import TensorType

from dino import modules

# See https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/dino.py
