
from __future__ import annotations

import numpy as np


class Perceptron:
    def __init__(self, input_dim=784, lr=0.01):
        if not isinstance(input_dim, (int, np.integer)) or isinstance(input_dim, (bool, np.bool_)) or input_dim <= 0:
            raise ValueError("input_dim 必须为正整数")
        if not np.isfinite(lr) or lr <= 0:
            raise ValueError("lr 必须为正的有限数")
        self.input_dim = int(input_dim)
        self.lr = float(lr)
        self.weights = np.zeros(self.input_dim, dtype=np.float64)
        self.bias = 0.0
        self.epoch_errors = []

    def _check_X(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X 必须是 (N,{self.input_dim})")
        if not np.issubdtype(X.dtype, np.number) or not np.isfinite(X).all():
            raise ValueError("X 必须是有限数值数组")
        return X

    def predict(self, X01) -> np.ndarray:
        """得分 >= 0 判为 1，其余判为 0，返回 (N,) 的 int64 数组。"""
        X = self._check_X(X01)
        return ((X @ self.weights + self.bias) >= 0).astype(np.int64)

    def fit(self, X01, y01, epochs) -> None:
        """每轮按原始顺序在线更新；错误更新量 = lr * (y - y_pred)。"""
        X = self._check_X(X01)
        y = np.asarray(y01)
        if y.ndim != 1 or len(y) != len(X) or len(y) == 0:
            raise ValueError("y01 必须是与 X01 等长的非空一维标签")
        if y.dtype.kind not in "iu" or not np.isin(y, [0, 1]).all():
            raise ValueError("y01 仅允许 0 或 1 的整数标签")
        if not isinstance(epochs, (int, np.integer)) or isinstance(epochs, (bool, np.bool_)) or epochs <= 0:
            raise ValueError("epochs 必须是正整数")
        self.epoch_errors = []
        for _ in range(epochs):
            mistakes = 0
            for xi, target in zip(X, y):
                pred = 1 if (float(xi @ self.weights) + self.bias) >= 0 else 0
                delta = self.lr * (int(target) - pred)
                if delta != 0:
                    self.weights += delta * xi
                    self.bias += delta
                    mistakes += 1
            self.epoch_errors.append(mistakes)
