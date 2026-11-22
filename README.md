# NumPy MNIST From Scratch

**基于 NumPy 手写神经网络训练框架，完成 MNIST 手写数字识别。**

本项目实现可配置的多层感知机（MLP），包括网络层、反向传播、损失函数、优化器、训练循环、模型保存与独立预测。另实现单层感知机，用于数字 0/1 二分类实验。

NumPy 用于数组和线性代数计算，Matplotlib 用于绘图。神经网络训练代码由小组自行实现，不使用现成训练框架或自动求导工具。

## 项目功能

- 自定义隐藏层层数与神经元数量
- 手写 Linear、ReLU 及反向传播
- 交叉熵与 MSE 损失函数
- SGD 与 Adam 优化器
- Accuracy、Macro-F1 等评估指标
- 保存 `.npz` 模型，并在独立程序中加载预测
- 单层感知机数字 0/1 分类实验
- 不同损失函数和优化器的对比实验

## 当前进度

| 模块 | 状态 |
| --- | --- |
| 公共接口、网络层与反向传播 | 已完成，数值梯度检查通过 |
| MNIST 数据读取与评估指标 | 已完成 |
| 交叉熵、MSE、SGD、Adam | 已完成，独立检查通过 |
| MLP 训练、模型保存与加载预测 | 已用真实 MNIST 验收 |
| 感知机独立演示与四组对比实验结果 | 正在整理 |

## 安装与准备数据

以下命令在项目根目录的 **Windows PowerShell** 中运行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_mnist.py --data-dir data
```

MNIST 数据将放在本地 `data/` 目录。

## 检查基础模块

```powershell
.\.venv\Scripts\python.exe -m src.check_layers
.\.venv\Scripts\python.exe -m checks.check_d
```

两项检查分别验证网络层的前向与反向梯度，以及损失函数和优化器。

## 训练模型

以下配置使用完整训练集训练 5 轮，并将模型保存到 `outputs/model_full.npz`：

```powershell
.\.venv\Scripts\python.exe -m src.train --data-dir data --output outputs/model_full.npz --epochs 5 --batch-size 64 --hidden 128 64 --optimizer adam --loss ce --lr 0.001
```

训练时观察每轮的 `train_loss`、`val_loss`、`val_acc` 和 `val_f1`。同目录下的 `model_full.json` 记录训练配置、逐轮指标和测试结果。

## 加载模型并预测

训练成功后，使用保存的模型评估完整测试集，并显示前 10 张图片的预测：

```powershell
.\.venv\Scripts\python.exe -m src.predict --data-dir data --model outputs/model_full.npz --split test --show 10
```

正式训练和完整测试均**不使用 `--limit`**。需要快速检查代码时，可以在训练命令中临时加入 `--limit 64`；该结果不作为正式实验结论。

## 已验证的基线结果

| 项目 | 结果 |
| --- | ---: |
| 数据划分 | 训练 54,000／验证 6,000／测试 10,000 |
| 网络结构 | 784 → 128 → 64 → 10 |
| 训练设置 | 交叉熵、Adam、学习率 0.001、批大小 64、5 轮 |
| 测试集 Accuracy | **97.31%** |
| 测试集 Macro-F1 | **97.29%** |

以上为一次真实 MNIST 运行结果。重新加载保存的模型后，得到相同的测试集指标。优化器与损失函数的完整对比结果将在实验材料中整理。

## 项目文档

- [接口约定](INTERFACES.md)：数据形状、模块函数及梯度规则
- [开发与复现说明](开发与复现说明.md)：成员分工、项目流程与交付安排
