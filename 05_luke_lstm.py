# -*- coding: utf-8 -*-
"""
05_luke_lstm.py —— 模块5 纵向时间序列验证（多任务 LSTM）
========================================================
用 Luke 单人 400 天日记数据，验证"压力/熬夜随时间变化是否导致脱发"，
并用一个多任务 LSTM 同时完成两个预测任务：
    1. 分类：会不会脱发（0/1）
    2. 回归：脱发严重度分数（1~4）

流程：数据加载 → 特征/目标编码 → 滑动窗口 → 按时间划分 → 训练 → 评估 → 可视化 → 保存模型。

约定：图表文字统一英文，中文仅用于控制台结论。
"""
import re
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, r2_score)
import joblib

# ----------------------------------------------------------------------------
# 路径与全局配置
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "reports" / "figures"
MODEL_DIR = BASE_DIR / "models"
FIG_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")
DEVICE = torch.device("cpu")

# 超参数
WINDOW = 7
HIDDEN_SIZE = 32
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 32
LAMBDA_REG = 0.5          # 回归损失权重 λ
SEED = 42

# 特征列（有序特征 + 数值特征）
ORDINAL_FEATURES = ["stress_level"]
NUMERIC_FEATURES = ["stay_up_late", "coffee_consumed", "brain_working_duration"]
FEATURES = ORDINAL_FEATURES + NUMERIC_FEATURES

STRESS_MAPPING = {"Low": 0, "Medium": 1, "High": 2, "Very High": 3}
HAIR_LOSS_CLS = {"Few": 0, "Medium": 1, "Many": 1, "A lot": 1}   # 会不会脱发
HAIR_LOSS_REG = {"Few": 1, "Medium": 2, "Many": 3, "A lot": 4}   # 严重度分数


def set_seed(seed: int = SEED) -> None:
    """固定随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一：小写、去除首尾空格、特殊字符转下划线。"""
    def _norm(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return re.sub(r"_+", "_", s).strip("_")
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# 数据加载与编码
# ----------------------------------------------------------------------------
def load_and_encode():
    """加载 Luke 数据并编码，返回特征矩阵、两类目标、日期序列。"""
    df = normalize_columns(pd.read_csv(DATA_DIR / "luke_hair_loss.csv"))

    dates = pd.to_datetime(df["date"], dayfirst=True).values

    # 特征矩阵
    X = np.zeros((len(df), len(FEATURES)), dtype=np.float32)
    for i, col in enumerate(FEATURES):
        if col in ORDINAL_FEATURES:
            X[:, i] = df[col].map(STRESS_MAPPING).values
        else:
            X[:, i] = df[col].astype(np.float32).values

    y_cls = df["hair_loss"].map(HAIR_LOSS_CLS).values.astype(np.float32)
    y_reg = df["hair_loss"].map(HAIR_LOSS_REG).values.astype(np.float32)

    return X, y_cls, y_reg, dates


def make_windows(X, y_cls, y_reg, N: int = WINDOW):
    """滑动窗口构造：用过去 N 天预测第 N+1 天。返回 (Xw, yc, yr)。"""
    Xs, yc, yr = [], [], []
    for i in range(len(X) - N):
        Xs.append(X[i:i + N])
        yc.append(y_cls[i + N])
        yr.append(y_reg[i + N])
    return np.array(Xs, dtype=np.float32), np.array(yc, dtype=np.float32), np.array(yr, dtype=np.float32)


class SeqDataset(Dataset):
    """序列数据集，供 DataLoader 使用。"""
    def __init__(self, X, yc, yr):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.yc = torch.tensor(yc, dtype=torch.float32)
        self.yr = torch.tensor(yr, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.yc[i], self.yr[i]


# ----------------------------------------------------------------------------
# 多任务 LSTM 模型
# ----------------------------------------------------------------------------
class MultiTaskLSTM(nn.Module):
    """共享 LSTM 编码器 + 分类头 + 回归头。"""
    def __init__(self, input_dim: int, hidden_size: int = HIDDEN_SIZE, dropout: float = DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.cls_head = nn.Linear(hidden_size, 1)   # 分类 logit
        self.reg_head = nn.Linear(hidden_size, 1)   # 回归输出

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # 取最后时间步的隐状态
        last = self.dropout(last)
        cls_logit = self.cls_head(last)
        reg_out = self.reg_head(last)
        return cls_logit, reg_out


# ----------------------------------------------------------------------------
# 训练
# ----------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR,
                patience=PATIENCE, lam=LAMBDA_REG):
    """训练多任务 LSTM，返回训练/验证损失曲线。"""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    best_val = float("inf")
    best_state = None
    counter = 0
    train_losses, val_losses = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for xb, ycb, yrb in train_loader:
            optimizer.zero_grad()
            cls_logit, reg_out = model(xb)
            loss = bce(cls_logit.squeeze(1), ycb) + lam * l1(reg_out.squeeze(1), yrb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, ycb, yrb in val_loader:
                cls_logit, reg_out = model(xb)
                loss = bce(cls_logit.squeeze(1), ycb) + lam * l1(reg_out.squeeze(1), yrb)
                val_loss += loss.item() * len(xb)
        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break

    model.load_state_dict(best_state)
    return train_losses, val_losses


# ----------------------------------------------------------------------------
# 评估
# ----------------------------------------------------------------------------
def predict(model, loader):
    """返回测试集的分类概率、分类预测、回归预测与真实值。"""
    model.eval()
    probas, reg_preds, yc_true, yr_true = [], [], [], []
    with torch.no_grad():
        for xb, ycb, yrb in loader:
            cls_logit, reg_out = model(xb)
            probas.append(torch.sigmoid(cls_logit).squeeze(1))
            reg_preds.append(reg_out.squeeze(1))
            yc_true.append(ycb)
            yr_true.append(yrb)
    probas = torch.cat(probas).numpy()
    reg_preds = torch.cat(reg_preds).numpy()
    yc_true = torch.cat(yc_true).numpy()
    yr_true = torch.cat(yr_true).numpy()
    cls_preds = (probas > 0.5).astype(int)
    return probas, cls_preds, reg_preds, yc_true, yr_true


def report_metrics(probas, cls_preds, reg_preds, yc_true, yr_true):
    """计算并打印分类与回归指标。"""
    print("=" * 70)
    print("多任务 LSTM 评估结果")
    print("=" * 70)

    acc = accuracy_score(yc_true, cls_preds)
    prec = precision_score(yc_true, cls_preds)
    rec = recall_score(yc_true, cls_preds)
    f1 = f1_score(yc_true, cls_preds)
    auc = roc_auc_score(yc_true, probas) if len(np.unique(yc_true)) > 1 else float("nan")

    print(f"[分类·会不会脱发] 准确率={acc:.4f} 精确率={prec:.4f} "
          f"召回率={rec:.4f} F1={f1:.4f} ROC-AUC={auc:.4f}")

    mae = mean_absolute_error(yr_true, reg_preds)
    rmse = np.sqrt(mean_squared_error(yr_true, reg_preds))
    r2 = r2_score(yr_true, reg_preds)

    print(f"[回归·严重度分数] MAE={mae:.4f} RMSE={rmse:.4f} R²={r2:.4f}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
            "roc_auc": auc, "mae": mae, "rmse": rmse, "r2": r2}


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def plot_loss(train_losses, val_losses):
    """训练/验证损失曲线。"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(train_losses, label="Train Loss")
    ax.plot(val_losses, label="Val Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Multi-task LSTM Training Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_lstm_loss_curve.png", dpi=300)
    plt.close(fig)


def plot_confusion(yc_true, cls_preds):
    """分类混淆矩阵。"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(yc_true, cls_preds)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix - LSTM Classification")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_lstm_confusion_matrix.png", dpi=300)
    plt.close(fig)


def plot_regression_scatter(yr_true, reg_preds):
    """回归：预测严重度 vs 真实严重度散点图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(yr_true, reg_preds, alpha=0.6, color="#4C72B0")
    lo, hi = 1, 4
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.5)
    ax.set_xlabel("Actual Severity Score")
    ax.set_ylabel("Predicted Severity Score")
    ax.set_title("Regression: Predicted vs Actual Severity")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_lstm_regression_scatter.png", dpi=300)
    plt.close(fig)


def plot_timeseries(dates, yr_true, reg_preds):
    """真实与预测严重度随日期变化的折线对比。"""
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(dates, yr_true, label="Actual Severity", marker="o", ms=3)
    ax.plot(dates, reg_preds, label="Predicted Severity", marker="o", ms=3, alpha=0.7)
    ax.set_xlabel("Date")
    ax.set_ylabel("Severity Score")
    ax.set_title("Actual vs Predicted Hair Loss Severity Over Time")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_lstm_timeseries.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    set_seed(SEED)

    X, y_cls, y_reg, dates = load_and_encode()
    print(f"Luke 数据: {len(X)} 天, 特征={FEATURES}")

    # 按时间划分：70% 训练 / 10% 验证 / 20% 测试（按天）
    n = len(X)
    tr_end = int(n * 0.7)
    va_end = int(n * 0.8)

    X_tr, X_va, X_te = X[:tr_end], X[tr_end:va_end], X[va_end:]
    yc_tr, yc_va, yc_te = y_cls[:tr_end], y_cls[tr_end:va_end], y_cls[va_end:]
    yr_tr, yr_va, yr_te = y_reg[:tr_end], y_reg[tr_end:va_end], y_reg[va_end:]

    # 标准化（仅用训练段拟合，避免数据泄漏）
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr).astype(np.float32)
    X_va = scaler.transform(X_va).astype(np.float32)
    X_te = scaler.transform(X_te).astype(np.float32)

    # 滑动窗口
    Xw_tr, yc_tr, yr_tr = make_windows(X_tr, yc_tr, yr_tr)
    Xw_va, yc_va, yr_va = make_windows(X_va, yc_va, yr_va)
    Xw_te, yc_te, yr_te = make_windows(X_te, yc_te, yr_te)
    print(f"样本数: 训练={len(Xw_tr)}, 验证={len(Xw_va)}, 测试={len(Xw_te)}")

    train_loader = DataLoader(SeqDataset(Xw_tr, yc_tr, yr_tr), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(Xw_va, yc_va, yr_va), batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(SeqDataset(Xw_te, yc_te, yr_te), batch_size=BATCH_SIZE, shuffle=False)

    model = MultiTaskLSTM(input_dim=len(FEATURES)).to(DEVICE)
    train_losses, val_losses = train_model(model, train_loader, val_loader)

    probas, cls_preds, reg_preds, yc_true, yr_true = predict(model, test_loader)
    metrics = report_metrics(probas, cls_preds, reg_preds, yc_true, yr_true)

    plot_loss(train_losses, val_losses)
    plot_confusion(yc_true, cls_preds)
    plot_regression_scatter(yr_true, reg_preds)

    # 测试段目标日期 = 测试段起始日 + WINDOW 之后
    test_target_dates = dates[va_end + WINDOW:]
    plot_timeseries(test_target_dates, yr_true, reg_preds)

    # 保存模型与配置
    torch.save(model.state_dict(), MODEL_DIR / "luke_lstm.pth")
    joblib.dump({
        "input_dim": len(FEATURES),
        "hidden_size": HIDDEN_SIZE,
        "window": WINDOW,
        "features": FEATURES,
        "stress_mapping": STRESS_MAPPING,
        "hair_loss_cls": HAIR_LOSS_CLS,
        "hair_loss_reg": HAIR_LOSS_REG,
        "scaler": scaler,
    }, MODEL_DIR / "luke_lstm_config.pkl")

    # 纵向结论
    print("\n" + "=" * 70)
    print("纵向验证结论")
    print("=" * 70)
    if not np.isnan(metrics["roc_auc"]) and metrics["roc_auc"] > 0.5:
        print(f"过去 {WINDOW} 天的压力/熬夜等序列可预测次日脱发（ROC-AUC={metrics['roc_auc']:.3f}），"
              "说明压力→脱发在时间维度成立。")
    else:
        print("模型未能从时间序列中有效预测脱发，提示 Luke 单一被试样本量不足。")

    print(f"\n模型已保存到: {MODEL_DIR}")
    print(f"图表已保存到: {FIG_DIR}")


if __name__ == "__main__":
    main()
