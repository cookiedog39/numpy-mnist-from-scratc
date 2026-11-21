"""在项目根目录执行 python -m src.check_layers，无需 MNIST 数据。"""

from __future__ import annotations

import numpy as np

from .core import Sequential
from .layers import Linear, ReLU


def numerical_gradient(fn, array: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """对标量函数逐元素做中心有限差分，计算后恢复原数组。"""
    grad = np.zeros_like(array)
    for index in np.ndindex(array.shape):
        original = array[index].copy()
        try:
            array[index] = original + eps
            plus = fn()
            array[index] = original - eps
            minus = fn()
        finally:
            array[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def check(name: str, analytical: np.ndarray, numerical: np.ndarray) -> None:
    np.testing.assert_allclose(analytical, numerical, rtol=1e-5, atol=1e-7)
    error = float(np.max(np.abs(analytical - numerical)))
    print(f"{name}: shape={analytical.shape}, max_abs_error={error:.3e} PASS")


def main() -> None:
    rng = np.random.default_rng(42)
    linear = Linear(3, 2, rng)
    x = rng.normal(size=(4, 3))
    dout = rng.normal(size=(4, 2))
    out = linear.forward(x)
    assert out.shape == (4, 2)
    grad_refs = [p.grad for p in linear.parameters()]
    dx = linear.backward(dout)
    dw, db = (p.grad.copy() for p in linear.parameters())
    objective = lambda: float(np.sum(linear.forward(x) * dout))
    check("Linear.dx", dx, numerical_gradient(objective, x))
    check("Linear.dW", dw, numerical_gradient(objective, linear.W.data))
    check("Linear.db", db, numerical_gradient(objective, linear.b.data))
    assert all(p.grad is ref for p, ref in zip(linear.parameters(), grad_refs))
    assert linear.parameters()[0] is linear.W
    assert linear.parameters()[1] is linear.b

    relu = ReLU()
    # 避开不可导的零点，且与有限差分步长保持足够距离。
    rx = np.array([[-1.2, 0.3, 2.0], [0.7, -0.5, -2.1]])
    rdout = rng.normal(size=rx.shape)
    relu.forward(rx)
    rdx = relu.backward(rdout)
    check("ReLU.dx", rdx, numerical_gradient(lambda: float(np.sum(relu.forward(rx) * rdout)), rx))
    assert relu.parameters() == []
    np.testing.assert_array_equal(relu.forward(np.array([-1., 0., 2.])), [0., 0., 2.])
    np.testing.assert_array_equal(relu.backward(np.ones(3)), [0., 0., 1.])

    model = Sequential([Linear(3, 4, rng), ReLU(), Linear(4, 2, rng)])
    mx = rng.normal(size=(2, 3))
    mdout = rng.normal(size=(2, 2))
    hidden = model.layers[0].forward(mx)
    assert np.min(np.abs(hidden)) > 1e-4
    model.forward(mx)
    mdx = model.backward(mdout)
    parameter_grads = [p.grad.copy() for p in model.parameters()]
    model_objective = lambda: float(np.sum(model.forward(mx) * mdout))
    check("Sequential.dx", mdx, numerical_gradient(model_objective, mx))
    for index, (param, grad) in enumerate(zip(model.parameters(), parameter_grads)):
        check(f"Sequential.p{index}", grad, numerical_gradient(model_objective, param.data))

    # 模拟尾批，确认可以改变批大小；验证 float32/float64 均可使用。
    for dtype in (np.float32, np.float64):
        for param in model.parameters():
            param.data = param.data.astype(dtype)
            param.grad = np.zeros_like(param.data)
        for batch_size in (4, 1):
            inputs = rng.normal(size=(batch_size, 3)).astype(dtype)
            outputs = model.forward(inputs)
            inputs_grad = model.backward(np.ones_like(outputs))
            assert outputs.shape == (batch_size, 2)
            assert inputs_grad.shape == inputs.shape
            assert outputs.dtype == dtype and inputs_grad.dtype == dtype
    print("Parameter references, ReLU at zero, variable batches and dtypes: PASS")
    print("All layer checks passed (rtol=1e-5, atol=1e-7).")


if __name__ == "__main__":
    main()
