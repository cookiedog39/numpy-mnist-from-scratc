
from __future__ import annotations

import gzip
from pathlib import Path
import struct

import numpy as np


def _find_idx_file(data_dir: Path, basename: str) -> Path:
    """允许将 MNIST 文件放在 data/、data/raw/ 或 data/MNIST/raw/。"""
    for folder in (data_dir, data_dir / "raw", data_dir / "MNIST" / "raw"):
        for name in (basename + ".gz", basename):
            path = folder / name
            if path.is_file():
                return path
    raise FileNotFoundError(
        f"找不到 {basename}(.gz)，请运行 python download_mnist.py --data-dir {data_dir}"
    )


def _read_idx(path: Path, is_image: bool) -> np.ndarray:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        header = f.read(16 if is_image else 8)
        if len(header) != (16 if is_image else 8):
            raise ValueError(f"IDX 文件头不完整: {path}")
        if is_image:
            magic, count, height, width = struct.unpack(">IIII", header)
            if magic != 2051 or (height, width) != (28, 28) or count == 0:
                raise ValueError(f"图像文件头异常（期望 magic=2051、28×28）: {path}")
            shape = (count, 28 * 28)
        else:
            magic, count = struct.unpack(">II", header)
            if magic != 2049 or count == 0:
                raise ValueError(f"标签文件头异常（期望 magic=2049）: {path}")
            shape = (count,)
        expected_size = int(np.prod(shape))
        payload = f.read()
        if len(payload) != expected_size:
            raise ValueError(
                f"IDX 数据长度异常: {path}，预期 {expected_size} 字节，实际 {len(payload)} 字节"
            )
        return np.frombuffer(payload, dtype=np.uint8).reshape(shape).copy()


def _read_pair(data_dir: Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    x = _read_idx(_find_idx_file(data_dir, f"{prefix}-images-idx3-ubyte"), True)
    y = _read_idx(_find_idx_file(data_dir, f"{prefix}-labels-idx1-ubyte"), False)
    if len(x) != len(y):
        raise ValueError(f"{prefix} 图片数和标签数不一致: {len(x)} vs {len(y)}")
    if np.any(y > 9):
        raise ValueError(f"{prefix} 中发现非 0～9 的标签")
    return x.astype(np.float32) / np.float32(255.0), y.astype(np.int64)


def load_mnist(data_dir, val_ratio=0.1, seed=42):
    """读取原始 MNIST；只从原训练集划出验证集，不动测试集。

    返回 ((X_train,y_train),(X_val,y_val),(X_test,y_test))。
    X: float32，(N,784)，范围 [0,1]；y: int64，(N,)。
    """
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须在 (0,1) 内")
    data_dir = Path(data_dir)
    X, y = _read_pair(data_dir, "train")
    X_test, y_test = _read_pair(data_dir, "t10k")
    if len(X) < 2:
        raise ValueError("训练集至少需要 2 张图片，才能划分训练/验证集")
    # 使用同一固定种子，方便全组复用相同的训练/验证划分。
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    n_val = max(1, min(len(y) - 1, int(round(len(y) * val_ratio))))
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return (X[train_idx], y[train_idx]), (X[val_idx], y[val_idx]), (X_test, y_test)


def iter_batches(X, y, batch_size, shuffle=False, seed=42):
    """逐批返回 (x_batch,y_batch)，包括最后一个不足 batch_size 的批次。"""
    X, y = np.asarray(X), np.asarray(y)
    if X.ndim != 2 or X.shape[1] != 784:
        raise ValueError("X 必须为 (N,784) 的二维数组")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y 必须为与 X 样本数相同的一维数组")
    if not isinstance(batch_size, (int, np.integer)) or isinstance(batch_size, (bool, np.bool_)) or batch_size <= 0:
        raise ValueError("batch_size 必须为正整数")
    indices = np.random.default_rng(seed).permutation(len(y)) if shuffle else np.arange(len(y))
    for start in range(0, len(y), batch_size):
        selected = indices[start : start + batch_size]
        yield X[selected], y[selected]
