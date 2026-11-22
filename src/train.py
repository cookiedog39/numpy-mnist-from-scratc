"""训练流程：按固定顺序执行清梯度、前向、损失、反向和参数更新。"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .checkpoint import save_model
from .losses import CrossEntropyLoss, MSELoss
from .model import build_mlp
from .optim import SGD, Adam

try:  # INTERFACES 约定在 src/ 下
    from .data import iter_batches, load_mnist
    from .metrics import accuracy, macro_f1
except ImportError:  # 当前仓库把 B 的模块放在项目根目录
    from data import iter_batches, load_mnist
    from metrics import accuracy, macro_f1


def make_loss(name: str):
    if name == "ce":
        return CrossEntropyLoss()
    if name == "mse":
        return MSELoss()
    raise ValueError("loss 只能是 ce 或 mse")


def make_optimizer(name: str, model, lr: float):
    params = model.parameters()
    if name == "sgd":
        return SGD(params, lr)
    if name == "adam":
        return Adam(params, lr)
    raise ValueError("optimizer 只能是 sgd 或 adam")


def evaluate(model, loss_fn, X, y, batch_size: int) -> dict:
    """只做前向和指标计算，不调用 backward / step。"""
    total_loss = 0.0
    predictions = []
    seen = 0
    for xb, yb in iter_batches(X, y, batch_size, shuffle=False):
        logits = model.forward(xb)
        total_loss += loss_fn.forward(logits, yb) * len(yb)
        predictions.append(np.argmax(logits, axis=1))
        seen += len(yb)
    pred = np.concatenate(predictions)
    return {
        "loss": total_loss / seen,
        "accuracy": accuracy(y, pred),
        "macro_f1": macro_f1(y, pred),
    }


def train_epoch(model, loss_fn, optimizer, X, y, batch_size: int, seed: int) -> float:
    total_loss = 0.0
    seen = 0
    for xb, yb in iter_batches(X, y, batch_size, shuffle=True, seed=seed):
        optimizer.zero_grad()
        logits = model.forward(xb)
        loss = loss_fn.forward(logits, yb)
        model.backward(loss_fn.backward())
        optimizer.step()
        total_loss += loss * len(yb)
        seen += len(yb)
    return total_loss / seen


def train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val,
          *, epochs: int, batch_size: int, seed: int = 42, log=print) -> list:
    history = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, loss_fn, optimizer, X_train, y_train,
                                 batch_size, seed + epoch)
        val = evaluate(model, loss_fn, X_val, y_val, batch_size)
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_macro_f1": val["macro_f1"],
        }
        history.append(record)
        log(f"epoch {epoch}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_loss={val['loss']:.4f}  val_acc={val['accuracy']:.4f}  "
            f"val_f1={val['macro_f1']:.4f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="训练 MNIST 手写 MLP")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs/model.npz")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--optimizer", choices=["sgd", "adam"], default="adam")
    parser.add_argument("--loss", choices=["ce", "mse"], default="ce")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 张训练，用于快速自检")
    args = parser.parse_args()

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_mnist(
        args.data_dir, val_ratio=args.val_ratio, seed=args.seed)
    if args.limit:
        X_train, y_train = X_train[: args.limit], y_train[: args.limit]

    model = build_mlp(tuple(args.hidden), seed=args.seed)
    loss_fn = make_loss(args.loss)
    optimizer = make_optimizer(args.optimizer, model, args.lr)

    print(f"训练集 {X_train.shape[0]}，验证集 {X_val.shape[0]}，测试集 {X_test.shape[0]}；"
          f"结构 784-{'-'.join(map(str, args.hidden))}-10；"
          f"{args.optimizer}+{args.loss}，lr={args.lr}，batch={args.batch_size}")
    started = time.time()
    history = train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed)
    elapsed = time.time() - started

    test = evaluate(model, loss_fn, X_test, y_test, args.batch_size)
    print(f"测试集 loss={test['loss']:.4f}  acc={test['accuracy']:.4f}  "
          f"macro_f1={test['macro_f1']:.4f}  用时 {elapsed:.1f}s")

    save_model(args.output, model, (784, *args.hidden, 10))
    history_path = Path(args.output).with_suffix(".json")
    history_path.write_text(json.dumps({"args": vars(args), "history": history, "test": test},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {args.output} 与 {history_path}")


if __name__ == "__main__":
    main()
