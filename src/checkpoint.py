"""模型保存与加载，格式为 .npz。

临时占位：本文件属 A（框架集成），A 交付后以他的版本为准。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .core import Sequential
from .model import build_mlp


def save_model(path, model: Sequential, layer_sizes) -> None:
    """保存层宽和按 parameters() 顺序编号的参数数组。"""
    sizes = tuple(int(size) for size in layer_sizes)
    params = model.parameters()
    arrays = {f"p{index}": param.data for index, param in enumerate(params)}
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target, layer_sizes=np.asarray(sizes, dtype=np.int64), **arrays)


def load_model(path) -> Sequential:
    """按层宽重建模型，并把参数原地写回。"""
    with np.load(Path(path), allow_pickle=False) as data:
        sizes = tuple(int(size) for size in data["layer_sizes"])
        model = build_mlp(sizes[1:-1], input_dim=sizes[0], num_classes=sizes[-1])
        params = model.parameters()
        keys = sorted((key for key in data.files if key.startswith("p")), key=lambda key: int(key[1:]))
        if len(keys) != len(params):
            raise ValueError(f"参数个数不一致：文件 {len(keys)}，模型 {len(params)}")
        for index, (key, param) in enumerate(zip(keys, params)):
            array = data[key]
            if array.shape != param.data.shape:
                raise ValueError(f"{key} 形状 {array.shape} 与模型 {param.data.shape} 不一致")
            param.data[...] = array
    return model
