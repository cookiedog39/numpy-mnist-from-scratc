"""对比实验：同一数据划分下比较 SGD/Adam 与交叉熵/MSE。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .model import build_mlp
from .train import evaluate, make_loss, make_optimizer, train

try:
    from .data import load_mnist
except ImportError:
    from data import load_mnist


COMBINATIONS = [
    ("ce", "sgd"),
    ("ce", "adam"),
    ("mse", "sgd"),
    ("mse", "adam"),
]


def run_one(loss_name, optimizer_name, lr, data, args, limit):
    (X_train, y_train), (X_val, y_val), _ = data
    if limit:
        X_train, y_train = X_train[:limit], y_train[:limit]
    model = build_mlp(tuple(args.hidden), seed=args.seed)
    loss_fn = make_loss(loss_name)
    optimizer = make_optimizer(optimizer_name, model, lr)
    history = train(model, loss_fn, optimizer, X_train, y_train, X_val, y_val,
                    epochs=args.epochs, batch_size=args.batch_size, seed=args.seed,
                    log=lambda message: None)
    final = history[-1]
    return model, loss_fn, history, final


def plot(results, target):
    epochs = [record["epoch"] for record in results[0]["history"]]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for item in results:
        label = f"{item['loss'].upper()} + {item['optimizer'].upper()}"
        axes[0].plot(epochs, [r["val_loss"] for r in item["history"]], marker="o", label=label)
        axes[1].plot(epochs, [r["val_accuracy"] for r in item["history"]], marker="o", label=label)
    axes[0].set_title("validation loss")
    axes[1].set_title("validation accuracy")
    for axis in axes:
        axis.set_xlabel("epoch")
        axis.grid(alpha=0.3)
        axis.legend()
    fig.tight_layout()
    fig.savefig(target, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SGD/Adam × 交叉熵/MSE 对比")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr-sgd", type=float, default=0.1)
    parser.add_argument("--lr-adam", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="只取前 N 张训练，用于快速自检")
    args = parser.parse_args()

    # 所有组合共用同一份划分与种子。
    data = load_mnist(args.data_dir, seed=args.seed)
    results = []
    for loss_name, optimizer_name in COMBINATIONS:
        lr = args.lr_sgd if optimizer_name == "sgd" else args.lr_adam
        print(f"=== {loss_name.upper()} + {optimizer_name.upper()}  lr={lr}")
        model, loss_fn, history, final = run_one(loss_name, optimizer_name, lr, data, args, args.limit)
        test = evaluate(model, loss_fn, data[2][0], data[2][1], args.batch_size)
        print(f"    val_acc={final['val_accuracy']:.4f}  val_f1={final['val_macro_f1']:.4f}  "
              f"test_acc={test['accuracy']:.4f}")
        results.append({
            "loss": loss_name,
            "optimizer": optimizer_name,
            "history": history,
            "test": test,
        })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot(results, output_dir / "experiments.png")
    summary_path = output_dir / "experiments.json"
    summary_path.write_text(json.dumps({"args": vars(args), "results": results},
                                       ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已保存 {output_dir / 'experiments.png'} 与 {summary_path}")


if __name__ == "__main__":
    main()
