"""Data augmentation and preprocessing functions."""
from __future__ import annotations

import math

import torch
from torch import nn
from torchtyping import TensorType

from dino import modules
