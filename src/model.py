"""按隐藏层配置组装 MLP。临时占位：本文件属 A，A 交付后以他的版本为准。"""

from __future__ import annotations

import numpy as np

from .core import Sequential
from .layers import Linear, ReLU


def build_mlp(hidden_sizes=(128, 64), input_dim=784, num_classes=10, seed=42) -> Sequential:
    """返回 输入 → [Linear, ReLU] × n → Linear 的 Sequential。"""
    for name, value in (("input_dim", input_dim), ("num_classes", num_classes)):
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    sizes = [int(input_dim)]
    for width in hidden_sizes:
        if not isinstance(width, (int, np.integer)) or isinstance(width, (bool, np.bool_)) or width <= 0:
            raise ValueError("hidden_sizes 必须全部为正整数")
        sizes.append(int(width))
    sizes.append(int(num_classes))

    rng = np.random.default_rng(seed)
    layers = []
    for index in range(len(sizes) - 1):
        layers.append(Linear(sizes[index], sizes[index + 1], rng=rng))
        if index < len(sizes) - 2:
            layers.append(ReLU())
    return Sequential(layers)
