"""神经网络的公共接口与顺序容器。

本文件只定义参数和层之间的连接方式。Linear、ReLU 等层的具体计算
由 layers.py 实现；损失和优化器分别由 losses.py、optim.py 实现。
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


class Parameter:
    """可学习参数及其梯度。

    data 和 grad 始终是同形状的 NumPy 浮点数组。保留传入的浮点精度：
    正式训练可用 float32，数值梯度检查可用 float64。
    """

    def __init__(self, data: np.ndarray) -> None:
        array = np.array(data, copy=True)
        if not np.issubdtype(array.dtype, np.floating):
            raise TypeError("Parameter.data 必须是 NumPy 浮点数组")
        self.data = array
        self.grad = np.zeros_like(array)

    def zero_grad(self) -> None:
        """原地清空梯度，不替换 grad 数组。"""
        self.grad.fill(0)


class Layer:
    """所有网络层共用的接口。"""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """计算本层输出；需要时由具体层保存反向传播所需的输入。"""
        raise NotImplementedError

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """写入本层参数的 grad，并返回损失对本层输入的梯度。"""
        raise NotImplementedError

    def parameters(self) -> list[Parameter]:
        """返回本层持有的 Parameter；无参数层返回空列表。"""
        return []


class Sequential(Layer):
    """将多个 Layer 按给定顺序连接成一个模型。"""

    def __init__(self, layers: Iterable[Layer]) -> None:
        self.layers = list(layers)
        for index, layer in enumerate(self.layers):
            if not isinstance(layer, Layer):
                raise TypeError(f"第 {index} 层必须继承 core.Layer")

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def parameters(self) -> list[Parameter]:
        params: list[Parameter] = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
