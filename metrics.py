
from __future__ import annotations

import numpy as np


def _labels(y_true, y_pred):
    truth, pred = np.asarray(y_true), np.asarray(y_pred)
    if truth.ndim != 1 or pred.ndim != 1 or len(truth) != len(pred):
        raise ValueError("y_true、y_pred 必须为长度相同的一维数组")
    if truth.size == 0:
        raise ValueError("指标计算不支持空输入")
    if truth.dtype.kind not in "iu" or pred.dtype.kind not in "iu":
        raise TypeError("标签必须是整数数组")
    return truth, pred


def accuracy(y_true, y_pred) -> float:
    """分类正确的样本数 / 总样本数。"""
    truth, pred = _labels(y_true, y_pred)
    return float(np.mean(truth == pred))


def confusion_matrix(y_true, y_pred, num_classes=10) -> np.ndarray:
    """返回 (num_classes,num_classes)，行真实，列预测。"""
    truth, pred = _labels(y_true, y_pred)
    if not isinstance(num_classes, (int, np.integer)) or isinstance(num_classes, (bool, np.bool_)) or num_classes <= 0:
        raise ValueError("num_classes 必须是正整数")
    if (np.any(truth < 0) or np.any(pred < 0)
            or np.any(truth >= num_classes) or np.any(pred >= num_classes)):
        raise ValueError("标签超出 [0,num_classes) 范围")
    encoded = truth.astype(np.int64) * num_classes + pred.astype(np.int64)
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)


def macro_f1(y_true, y_pred, num_classes=10) -> float:
    """所有指定类别 F1 的算术平均；零分母类别贡献 0。"""
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    tp = np.diag(cm).astype(np.float64)
    actual = cm.sum(axis=1).astype(np.float64)
    predicted = cm.sum(axis=0).astype(np.float64)
    denominator = actual + predicted  # == 2*TP + FP + FN
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    return float(np.mean(f1))
