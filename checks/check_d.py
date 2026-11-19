"""在项目根目录运行 python -m checks.check_d，独立验收损失函数与优化器。"""

import numpy as np

from src.core import Parameter
from src.losses import CrossEntropyLoss, MSELoss
from src.optim import Adam, SGD


def check_loss(loss_type):
    loss = loss_type()
    logits = np.array([[0.3, -0.2, 0.7], [-0.4, 0.6, 0.1]])
    y = np.array([2, 0], dtype=np.int64)
    loss.forward(logits, y)
    analytic = loss.backward()
    numeric = np.empty_like(logits)
    h = 1e-5
    for index in np.ndindex(logits.shape):
        offset = np.zeros_like(logits)
        offset[index] = h
        numeric[index] = (loss.forward(logits + offset, y)
                          - loss.forward(logits - offset, y)) / (2 * h)
    np.testing.assert_allclose(analytic, numeric, atol=1e-8, rtol=0)
    print(f"{loss_type.__name__} gradient max error = {np.max(np.abs(analytic - numeric)):.2e}")

    # 单样本末批：十类等分时交叉熵为 ln(10)，MSE 为 1/10。
    scores = np.zeros((1, 10), dtype=np.float32)
    expected = np.log(10) if loss_type is CrossEntropyLoss else 0.1
    np.testing.assert_allclose(loss.forward(scores, np.array([7])), expected)
    assert loss.backward().shape == scores.shape
    assert loss.backward().dtype == scores.dtype


def check_stability():
    loss = CrossEntropyLoss()
    scores = np.array([[1000, -1000], [-1000, 1000]], dtype=np.float32)
    value = loss.forward(scores, np.array([1, 0]))
    np.testing.assert_allclose(value, 2000)
    np.testing.assert_allclose(loss.backward(), [[0.5, -0.5], [-0.5, 0.5]])
    print(f"Extreme logits: CE = {value:.1f}, gradients finite")


def check_updates(optimizer_type):
    p = Parameter(np.array([0.0, 0.0]))
    optimizer = optimizer_type([p], lr=0.1)
    data, grad = p.data, p.grad
    for sign in [1, -1]:
        optimizer.zero_grad()
        np.testing.assert_array_equal(p.grad, [0, 0])
        p.grad[:] = [sign, -sign]
        optimizer.step()
        assert data is p.data and grad is p.grad
    # 反向等幅梯度使 SGD 回到起点；Adam 第二步的一阶矩估计为 -1/19。
    expected = 0 if optimizer_type is SGD else -0.1 * (1 - 1 / 19) / (1 + 1e-8)
    np.testing.assert_allclose(p.data, [expected, -expected], atol=1e-12, rtol=0)


def check_training(loss_type, optimizer_type):
    # 固定的三条特征和十个输出类别；用仿射模型验证 W、b 的完整更新路径。
    x = np.eye(3, dtype=np.float32)
    y = np.array([2, 5, 8], dtype=np.int64)
    w = Parameter(np.zeros((3, 10), dtype=np.float32))
    b = Parameter(np.zeros(10, dtype=np.float32))
    loss = loss_type()
    optimizer = optimizer_type([w, b], lr=0.05)
    initial = loss.forward(x @ w.data + b.data, y)
    for _ in range(10):
        optimizer.zero_grad()
        loss.forward(x @ w.data + b.data, y)
        dout = loss.backward()
        w.grad[:] = x.T @ dout
        b.grad[:] = dout.sum(axis=0)
        optimizer.step()
    final = loss.forward(x @ w.data + b.data, y)
    assert np.isfinite(final) and final < initial
    print(f"{loss_type.__name__:16} + {optimizer_type.__name__:4}  {initial:.6f} -> {final:.6f}")


if __name__ == "__main__":
    for loss_type in (CrossEntropyLoss, MSELoss):
        check_loss(loss_type)
    check_stability()
    for optimizer_type in (SGD, Adam):
        check_updates(optimizer_type)
        for loss_type in (CrossEntropyLoss, MSELoss):
            check_training(loss_type, optimizer_type)
    print("All D checks passed.")
