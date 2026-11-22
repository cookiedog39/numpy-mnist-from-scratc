"""加载模型并对图片输出 0～9 的预测。"""

from __future__ import annotations

import argparse

import numpy as np

from .checkpoint import load_model

try:
    from .data import load_mnist
    from .metrics import accuracy, macro_f1
except ImportError:
    from data import load_mnist
    from metrics import accuracy, macro_f1


def predict(model, X, batch_size: int = 512) -> np.ndarray:
    """分批前向，返回 (N,) 的预测标签。"""
    labels = [
        np.argmax(model.forward(X[start:start + batch_size]), axis=1)
        for start in range(0, len(X), batch_size)
    ]
    return np.concatenate(labels) if labels else np.empty(0, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="加载 .npz 模型并在 MNIST 上预测")
    parser.add_argument("--model", default="outputs/model.npz")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--limit", type=int, default=0, help="只预测前 N 张")
    parser.add_argument("--show", type=int, default=10, help="打印前 N 条的预测与真值")
    args = parser.parse_args()

    _, (X_val, y_val), (X_test, y_test) = load_mnist(args.data_dir)
    X, y = (X_test, y_test) if args.split == "test" else (X_val, y_val)
    if args.limit:
        X, y = X[: args.limit], y[: args.limit]

    model = load_model(args.model)
    pred = predict(model, X)
    print(f"{args.split} 集 {len(y)} 张：acc={accuracy(y, pred):.4f}  macro_f1={macro_f1(y, pred):.4f}")
    for index in range(min(args.show, len(y))):
        print(f"  第 {index + 1} 张  预测 {pred[index]}  真值 {y[index]}")


if __name__ == "__main__":
    main()
