"""MLP 分类损失。输入为原始分数和整数标签，梯度按实际批大小缩放。"""

import numpy as np


def _check_batch(logits: np.ndarray, y: np.ndarray) -> None:
    if logits.ndim != 2 or 0 in logits.shape:
        raise ValueError("logits 必须是非空的 (B, C) 数组")
    if not np.issubdtype(logits.dtype, np.floating):
        raise TypeError("logits 必须是浮点数组")
    if y.shape != (logits.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("y 必须是形状为 (B,) 的整数标签")
    if np.any(y < 0) or np.any(y >= logits.shape[1]):
        raise ValueError("标签必须在 [0, C) 范围内")


class CrossEntropyLoss:
    """对每行分数计算稳定的 log-sum-exp，再减去正确类分数。"""

    def __init__(self) -> None:
        self._residual = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        _check_batch(logits, y)
        rows = np.arange(y.size)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        normalizer = exp_scores.sum(axis=1, keepdims=True)
        losses = np.log(normalizer[:, 0]) - shifted[rows, y]

        # 在正确类别位置减 1，等价于减去 one-hot 标签。
        self._residual = exp_scores / normalizer
        self._residual[rows, y] -= 1
        return float(losses.mean(dtype=np.float64))

    def backward(self) -> np.ndarray:
        if self._residual is None:
            raise RuntimeError("请先调用 forward")
        return self._residual / self._residual.shape[0]


class MSELoss:
    """将整数标签视为 one-hot 目标，对全部 B × C 个平方误差取均值。"""

    def __init__(self) -> None:
        self._residual = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        _check_batch(logits, y)
        self._residual = logits.copy()
        self._residual[np.arange(y.size), y] -= 1
        return float(np.square(self._residual, dtype=np.float64).mean())

    def backward(self) -> np.ndarray:
        if self._residual is None:
            raise RuntimeError("请先调用 forward")
        return self._residual * (2.0 / self._residual.size)
