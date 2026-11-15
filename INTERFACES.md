# MNIST 手写训练框架：接口约定

## 1. 项目边界

- 主任务：使用 NumPy 手写可配置的多层感知机（MLP），完成 MNIST 0～9 分类。
- 独立必选实验：用感知机学习规则完成 MNIST 数字 0/1 二分类。
- 自行实现网络层、反向传播、损失函数、优化器和训练循环。NumPy 用于数组与线性代数计算；Matplotlib 仅用于绘图。不引入现成神经网络训练框架、自动求导或现成训练器。
- 默认网络结构为 `784 → 128 → 64 → 10`，隐藏层层数和宽度由 `hidden_sizes` 控制。

## 2. 全局数据约定

| 名称 | 类型与形状 | 含义 |
| --- | --- | --- |
| `X` | `np.ndarray`，训练时 `float32`，`(B, 784)` | 每行一张展平的 28×28 图片；像素值在 `[0, 1]` |
| `y` | `np.ndarray`，`int64`，`(B,)` | 每张图片的整数标签，取值 0～9 |
| `logits` | `np.ndarray`，`(B, 10)` | 模型输出的十类原始分数，尚未做 Softmax |
| `dlogits` | `np.ndarray`，与 `logits` 同形 | 损失对原始分数的梯度 |

`B` 表示当前批次的样本数，最后一批可以小于设定的批大小。数据模块不做 one-hot；模型最后一层不加 Softmax。交叉熵在损失函数内部完成数值稳定的 Softmax 相关计算。数值梯度检查可以临时使用 `float64`，实现不得强制把所有输入转为 `float32`。

## 3. A：公共层、模型组装和保存加载

### `src/core.py`

```python
Parameter(data)
# 属性：data 为参数数组；grad 与 data 同形，初始为 0。

Layer.forward(x) -> out
Layer.backward(dout) -> dx
Layer.parameters() -> list[Parameter]

Sequential(layers)
# 提供与 Layer 相同的三个方法。
```

`Sequential.forward` 按层顺序调用；`Sequential.backward` 按相反顺序调用。每层的 `backward` 将本层参数梯度写入对应 `Parameter.grad`，并返回传给前一层的 `dx`。`parameters()` 按层顺序返回**原始 Parameter 对象的引用**，顺序必须稳定。

### `src/model.py`

```python
build_mlp(hidden_sizes=(128, 64), input_dim=784,
          num_classes=10, seed=42) -> Sequential
```

用同一个 `np.random.Generator` 初始化各个 `Linear` 层；在每个隐藏层的 `Linear` 后加入 `ReLU`，最后一层 `Linear` 后不加激活或 Softmax。`hidden_sizes` 中的值均须为正整数，允许使用不同的隐藏层层数。

### `src/checkpoint.py`

```python
save_model(path, model, layer_sizes) -> None
load_model(path) -> Sequential
```

`layer_sizes` 例如 `(784, 128, 64, 10)`。保存为 `.npz`：包含 `layer_sizes` 以及按 `model.parameters()` 稳定顺序编号的 `p0, p1, ...`。加载时使用 `np.load(..., allow_pickle=False)`，按层宽重建模型，核对参数数量和形状，再将数组**原地写入** `Parameter.data`。当前只需保存用于预测的模型；继续训练所需的优化器状态不在第一版范围内。

## 4. B：数据、指标和独立感知机

### `src/data.py`

```python
load_mnist(data_dir, val_ratio=0.1, seed=42)
    -> ((X_train, y_train), (X_val, y_val), (X_test, y_test))

iter_batches(X, y, batch_size, shuffle=False, seed=42)
    -> 逐批产生 (x_batch, y_batch)
```

验证集只从 MNIST 原训练集划分，原测试集保留作最终测试。划分和打乱使用传入的 `seed`，保证可复现；`iter_batches` 不丢弃最后一个不足批大小的批次。

### `src/metrics.py`

```python
accuracy(y_true, y_pred) -> float
macro_f1(y_true, y_pred, num_classes=10) -> float
confusion_matrix(y_true, y_pred, num_classes=10) -> np.ndarray
```

`y_true`、`y_pred` 均为一维整数标签。`accuracy` 和 `macro_f1` 返回 `[0, 1]` 范围内的数值；混淆矩阵形状为 `(num_classes, num_classes)`，行是真实标签、列是预测标签。Macro-F1 对 `0 ... num_classes-1` 各类的 F1 取算术平均；某类分母为 0 时该类的精确率/召回率/F1 记为 0。

### `src/perceptron.py`

```python
Perceptron(input_dim=784, lr=0.01)
Perceptron.fit(X01, y01, epochs) -> None
Perceptron.predict(X01) -> np.ndarray
```

`X01` 为 `(B, 784)`；`y01` 与预测输出均为形状 `(B,)`、取值 0 或 1 的整数标签。感知机学习规则在该模块独立实现，不复用 MLP 的反向传播与优化器。

## 5. C：网络层与手写反向传播

### `src/layers.py`

```python
Linear(in_features, out_features, rng=None)
ReLU()
# 两者均实现 Layer.forward / backward / parameters。
```

`Linear` 使用 `W.shape == (in_features, out_features)`、`b.shape == (out_features,)`，输入 `(B, in_features)`、输出 `(B, out_features)`。`ReLU` 保持输入输出形状不变且无可学习参数。层在前向时保存反向所需的中间结果。C 须提供小规模有限差分梯度检查的运行脚本与结果。

`Linear`、`ReLU` 均须继承 `src.core.Layer`；可学习参数统一使用 `src.core.Parameter`，不要在 `layers.py` 中另定义一套参数类。

**梯度缩放规则：**损失函数的 `backward` 负责按批次平均；`Linear.backward` 按链式法则计算 `dW = X.T @ dout`、`db = dout.sum(axis=0)`、`dx = dout @ W.T`，不得再次除以 `B`。

## 6. D：损失与优化器

### `src/losses.py`

```python
CrossEntropyLoss.forward(logits, y) -> float
CrossEntropyLoss.backward() -> dlogits

MSELoss.forward(logits, y) -> float
MSELoss.backward() -> dlogits
```

交叉熵接收原始 `logits`，使用数值稳定算法，损失按样本数 `B` 平均；其 `backward` 输出 `(softmax(logits) - one_hot(y)) / B`。MSE 在内部将标签转为 one-hot，与原始 `logits` 比较；若 `forward` 定义为所有 `B×10` 个元素的均方值，则 `backward` 为 `2 × (logits - one_hot(y)) / (B×10)`。两种损失的前向与反向必须使用同一套定义。

### `src/optim.py`

```python
SGD(params, lr)
Adam(params, lr)
# 两者均提供 zero_grad() -> None、step() -> None。
```

`params` 是 `model.parameters()` 返回的稳定参数列表。`zero_grad` 原地清零各参数的 `.grad`；`step` 根据梯度原地更新 `.data`。优化器在一次训练任务开始时创建一次，不能每批重新创建；Adam 的历史状态必须跨批保留。

## 7. E：训练、实验与预测

### `src/train.py`

每批训练固定调用顺序：

```python
optimizer.zero_grad()
logits = model.forward(x_batch)
train_loss = loss_fn.forward(logits, y_batch)
model.backward(loss_fn.backward())
optimizer.step()
```

验证/测试阶段只执行前向和指标计算，不调用 `backward` 或 `step`。训练日志至少记录 epoch、训练损失、验证损失、验证 Accuracy 和 Macro-F1。测试集只用于最终结果。

### `src/predict.py` 与 `src/experiments.py`

预测入口调用 `load_model`，对输入执行与训练一致的预处理，再以 `np.argmax(logits, axis=1)` 输出 0～9 的预测标签。实验脚本至少比较 SGD/Adam 和交叉熵/MSE，并固定数据划分、网络结构、随机种子及训练轮数，记录各实验配置和实际结果。

## 8. 模块交付与变更规则

每位编码成员提交：**源码、最小运行命令、实际输出或日志、半页技术说明**。首次交接时先用小数组验证接口，再进入完整 MNIST 训练。任何公共签名、形状或返回值的变更，先说明受影响模块，由 A 更新本文件并通知全组。

