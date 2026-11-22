"""使用层写入的 Parameter.grad 原地更新参数。"""

from collections.abc import Iterable

import numpy as np

from .core import Parameter


class _Optimizer:
    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()


class SGD(_Optimizer):
    """小批量随机梯度下降。"""

    def step(self) -> None:
        for param in self.params:
            param.data -= self.lr * param.grad


class Adam(_Optimizer):
    """保留每个参数的一阶矩、二阶矩，并按更新步数校正初始化偏差。"""

    def __init__(self, params: Iterable[Parameter], lr: float) -> None:
        super().__init__(params, lr)
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self) -> None:
        self.t += 1
        for param, m, v in zip(self.params, self.m, self.v):
            m *= self.beta1
            m += (1 - self.beta1) * param.grad
            v *= self.beta2
            v += (1 - self.beta2) * np.square(param.grad)
            m_hat = m / (1 - self.beta1 ** self.t)
            v_hat = v / (1 - self.beta2 ** self.t)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
