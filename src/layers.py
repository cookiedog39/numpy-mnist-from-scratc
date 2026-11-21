"""MLP 的全连接层、ReLU 激活及手写反向传播。"""

from __future__ import annotations

import numpy as np

from .core import Layer, Parameter


class Linear(Layer):
    """计算 X @ W + b；参数顺序固定为 W、b。

    使用 He 正态初始化和零偏置。默认保留 NumPy 的 float64 精度，
    不强制转换输入精度，便于有限差分检查。反向传播覆盖本层梯度，
    不在层内做批次平均；平均由损失函数负责。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        for name, value in (("in_features", in_features), ("out_features", out_features)):
            if (
                not isinstance(value, (int, np.integer))
                or isinstance(value, (bool, np.bool_))
                or value <= 0
            ):
                raise ValueError(f"{name} 必须为正整数")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        rng = np.random.default_rng() if rng is None else rng
        self.W = Parameter(
            rng.standard_normal((self.in_features, self.out_features))
            * np.sqrt(2.0 / self.in_features)
        )
        self.b = Parameter(np.zeros(self.out_features, dtype=self.W.data.dtype))
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(f"Linear 输入形状必须为 (B, {self.in_features})")
        self._x = x
        return x @ self.W.data + self.b.data

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("Linear.backward 前必须先调用 forward")
        dout = np.asarray(dout)
        if dout.shape != (self._x.shape[0], self.out_features):
            raise ValueError("Linear 上游梯度形状必须与最近一次前向输出一致")
        # 原地写入，保留优化器持有的梯度数组引用。
        self.W.grad[...] = self._x.T @ dout
        self.b.grad[...] = dout.sum(axis=0)
        return dout @ self.W.data.T

    def parameters(self) -> list[Parameter]:
        return [self.W, self.b]


class ReLU(Layer):
    """逐元素取 max(0, x)，约定 x == 0 时导数为 0。"""

    def __init__(self) -> None:
        self._positive: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        self._positive = x > 0
        return np.maximum(x, 0)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if self._positive is None:
            raise RuntimeError("ReLU.backward 前必须先调用 forward")
        dout = np.asarray(dout)
        if dout.shape != self._positive.shape:
            raise ValueError("ReLU 上游梯度形状必须与最近一次前向输出一致")
        return dout * self._positive

    def parameters(self) -> list[Parameter]:
        return []
