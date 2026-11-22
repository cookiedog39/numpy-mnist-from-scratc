"""下载 MNIST 原始文件到 data/，供 data.py 读取。

用法：python download_mnist.py --data-dir data
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

MIRRORS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
]

# 文件名 -> 期望字节数，用来发现被截断的下载
FILES = {
    "train-images-idx3-ubyte.gz": 9912422,
    "train-labels-idx1-ubyte.gz": 28881,
    "t10k-images-idx3-ubyte.gz": 1648877,
    "t10k-labels-idx1-ubyte.gz": 4542,
}


def download_one(name: str, size: int, target: Path) -> None:
    last_error = None
    for mirror in MIRRORS:
        try:
            urllib.request.urlretrieve(mirror + name, target)
        except Exception as error:  # 换下一个镜像
            last_error = error
            continue
        if target.stat().st_size == size:
            return
        last_error = ValueError(f"{name} 大小异常：{target.stat().st_size}，预期 {size}")
        target.unlink()
    raise SystemExit(f"下载失败：{name}（{last_error}）")


def main() -> None:
    parser = argparse.ArgumentParser(description="下载 MNIST 四个 idx 文件")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--force", action="store_true", help="已存在时重新下载")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, size in FILES.items():
        target = data_dir / name
        if target.is_file() and not args.force:
            print(f"已存在，跳过 {target}")
            continue
        print(f"下载 {name} …")
        download_one(name, size, target)
        print(f"完成 {target}（{target.stat().st_size} 字节）")


if __name__ == "__main__":
    main()
