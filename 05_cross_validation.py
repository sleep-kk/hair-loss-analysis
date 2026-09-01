# -*- coding: utf-8 -*-
"""
05_cross_validation.py —— 模块5 跨数据集验证（结论是否可靠）
============================================================
包含三个子任务：
    1. 单因素结论一致性对比（多源三角验证）
    2. 跨数据集模型迁移（有限度，诚实呈现迁移能力）
    3. Cohen's Kappa 一致性系数

约定：图表文字统一英文，中文仅用于控制台结论。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind, chi2_contingency, spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (cohen_kappa_score, roc_auc_score, accuracy_score, roc_curve)

# ----------------------------------------------------------------------------
# 路径与全局配置
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

DATASET_ORDER = ["Mendeley", "Kaggle", "UCI", "Luke"]
FACTOR_ORDER = ["Age", "Gender", "Stress", "Family History"]


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一：小写、去空格、特殊字符转下划线。"""
    def _norm(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return re.sub(r"_+", "_", s).strip("_")
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


# ----------------------------------------------------------------------------
# 数据加载与清洗
# ----------------------------------------------------------------------------
def load_datasets() -> dict:
    """加载并清洗 4 个数据集。"""
    m = normalize_columns(pd.read_csv(DATA_DIR / "mendeley_hair_loss_survey.csv"))
    m = m.replace({"Yea": "Yes", "\\No": "No"})
    m = m[m["what_is_your_age"] <= 100]

    k = normalize_columns(pd.read_csv(DATA_DIR / "kaggle_hair_health.csv"))

    u = normalize_columns(pd.read_csv(DATA_DIR / "uci_diabetes.csv"))
    u = u.drop_duplicates()

    l = normalize_columns(pd.read_csv(DATA_DIR / "luke_hair_loss.csv"))

    return {"Mendeley": m, "Kaggle": k, "UCI": u, "Luke": l}


def binarize_target(dataset: str, series: pd.Series) -> pd.Series:
    """目标二值化：1=脱发/严重脱发，0=否。"""
    s = series.astype(str).str.strip()
    if dataset == "Luke":
        return s.isin(["Many", "A lot"]).astype(int)   # 严重脱发
    return s.isin(["Yes", "1", "1.0"]).astype(int)


# ----------------------------------------------------------------------------
# 单因素一致性对比
# ----------------------------------------------------------------------------
# (因素名, 数据集, 因素列, 目标列, 因素类型 continuous/categorical)
COMPARISONS = [
    ("Age", "Mendeley", "what_is_your_age", "do_you_have_hair_fall_problem", "continuous"),
    ("Age", "Kaggle", "age", "hair_loss", "continuous"),
    ("Age", "UCI", "age", "alopecia", "continuous"),
    ("Gender", "Mendeley", "what_is_your_gender", "do_you_have_hair_fall_problem", "categorical"),
    ("Gender", "UCI", "gender", "alopecia", "categorical"),
    ("Stress", "Mendeley", "do_you_have_too_much_stress", "do_you_have_hair_fall_problem", "categorical"),
    ("Stress", "Kaggle", "stress", "hair_loss", "categorical"),
    ("Stress", "Luke", "stress_level", "hair_loss", "categorical"),
    ("Family History", "Mendeley", "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
     "do_you_have_hair_fall_problem", "categorical"),
    ("Family History", "Kaggle", "genetics", "hair_loss", "categorical"),
]


def categorical_direction(factor_name: str, f: pd.Series, y: pd.Series) -> int:
    """计算分类因素的方向符号（+1 正向、-1 负向）。"""
    rates = y.groupby(f).mean()
    if factor_name == "Gender":
        return 1 if rates.get("Male", 0) >= rates.get("Female", 0) else -1
    if factor_name == "Stress":
        uniq = set(f.astype(str))
        if uniq <= {"Yes", "No"}:
            return 1 if rates.get("Yes", 0) >= rates.get("No", 0) else -1
        order = {"Low": 0, "Medium": 1, "Moderate": 1, "High": 2, "Very High": 3}
        ranks = f.map(lambda v: order.get(v, 0))
        rho, _ = spearmanr(ranks, y)
        return 1 if rho >= 0 else -1
    # Family History 等二值 Yes/No
    return 1 if rates.get("Yes", 0) >= rates.get("No", 0) else -1


def analyze_one(comp, datasets) -> dict:
    """对单个 (因素, 数据集) 组合做检验，返回结论记录。"""
    factor, dataset, fcol, tcol, ftype = comp
    df = datasets[dataset]
    y = binarize_target(dataset, df[tcol])
    f = df[fcol]
    valid = f.notna() & y.notna()
    f, y = f[valid], y[valid]

    if ftype == "continuous":
        f = f.astype(float)
        g1, g0 = f[y == 1], f[y == 0]
        _, p = ttest_ind(g1, g0)
        direction = 1 if g1.mean() >= g0.mean() else -1
        detail = f"有脱发 {g1.mean():.1f} vs 无 {g0.mean():.1f}"
    else:
        _, p, _, _ = chi2_contingency(pd.crosstab(f, y))
        direction = categorical_direction(factor, f, y)
        rates = y.groupby(f).mean()
        detail = ", ".join(f"{k}={v*100:.0f}%" for k, v in rates.items())

    sig = p < 0.05
    sign = direction if sig else 0
    label = "+" if sign == 1 else ("-" if sign == -1 else "ns")
    return {"factor": factor, "dataset": dataset, "p": p, "sig": sig,
            "direction": direction, "sign": sign, "label": label, "detail": detail}


def consistency_analysis(datasets) -> pd.DataFrame:
    """运行所有单因素比较，返回一致性表。"""
    records = [analyze_one(c, datasets) for c in COMPARISONS]
    table = pd.DataFrame(records)
    return table


def plot_consistency_heatmap(table: pd.DataFrame) -> None:
    """一致性热力图：行=因素，列=数据集，单元格=显著方向。"""
    pivot = table.pivot_table(index="factor", columns="dataset", values="sign",
                              aggfunc="first").reindex(FACTOR_ORDER, columns=DATASET_ORDER)
    label_pivot = table.pivot_table(index="factor", columns="dataset", values="label",
                                    aggfunc="first").reindex(FACTOR_ORDER, columns=DATASET_ORDER)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.heatmap(pivot, annot=label_pivot, fmt="", cmap="coolwarm", center=0,
                vmin=-1, vmax=1, cbar_kws={"label": "Direction (significant)"}, ax=ax)
    ax.set_title("Cross-dataset Consistency of Single-factor Effects")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Factor")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_consistency_heatmap.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Cohen's Kappa
# ----------------------------------------------------------------------------
def compute_kappa(table: pd.DataFrame) -> None:
    """对数据集两两计算"因素显著性标签"的 Cohen's Kappa。"""
    print("\n" + "=" * 70)
    print("Cohen's Kappa 一致性系数（基于各因素在数据集中的显著性标签）")
    print("=" * 70)
    sig_pivot = table.pivot_table(index="factor", columns="dataset", values="sig",
                                  aggfunc="first").reindex(columns=DATASET_ORDER)

    pairs = [("Mendeley", "Kaggle"), ("Mendeley", "UCI"), ("Kaggle", "UCI"),
             ("Mendeley", "Luke"), ("Kaggle", "Luke"), ("UCI", "Luke")]
    for a, b in pairs:
        sub = sig_pivot[[a, b]].dropna().astype(int)
        if len(sub) < 2:
            print(f"  {a} vs {b}: 共同因素不足（{len(sub)}），无法可靠计算 Kappa")
            continue
        la, lb = sub[a].values, sub[b].values
        if len(np.unique(np.concatenate([la, lb]))) < 2:
            state = "均显著" if la[0] == 1 else "均不显著"
            print(f"  {a} vs {b}: 共同因素={len(sub)} 个，显著性标签完全一致（{state}），Kappa 无定义")
            continue
        kappa = cohen_kappa_score(la, lb)
        print(f"  {a} vs {b}: 共同因素={len(sub)} 个, Cohen's Kappa = {kappa:.3f}")


# ----------------------------------------------------------------------------
# 跨数据集迁移
# ----------------------------------------------------------------------------
def transfer_experiment(m: pd.DataFrame, k: pd.DataFrame) -> None:
    """用 Mendeley(年龄+压力) 训练逻辑回归，迁移到 Kaggle 测试。"""
    print("\n" + "=" * 70)
    print("跨数据集迁移实验：Mendeley(年龄+压力) → Kaggle")
    print("=" * 70)

    # 源域 Mendeley
    mf = pd.DataFrame({
        "age": m["what_is_your_age"].astype(float),
        "stress": m["do_you_have_too_much_stress"].map({"Yes": 1, "No": 0}),
        "y": m["do_you_have_hair_fall_problem"].map({"Yes": 1, "No": 0}),
    }).dropna()
    Xm = mf[["age", "stress"]].values
    ym = mf["y"].values.astype(int)

    # 目标域 Kaggle（压力映射：High->1, 其余->0）
    kf = pd.DataFrame({
        "age": k["age"].astype(float),
        "stress": k["stress"].map({"High": 1, "Moderate": 0, "Low": 0}),
        "y": k["hair_loss"].astype(int),
    }).dropna()
    Xk = kf[["age", "stress"]].values
    yk = kf["y"].values.astype(int)

    # 标准化年龄（仅用源域拟合）
    scaler = StandardScaler()
    Xm[:, 0] = scaler.fit_transform(Xm[:, 0].reshape(-1, 1)).ravel()
    Xk[:, 0] = scaler.transform(Xk[:, 0].reshape(-1, 1)).ravel()

    model = LogisticRegression(max_iter=1000)
    model.fit(Xm, ym)
    proba = model.predict_proba(Xk)[:, 1]
    pred = model.predict(Xk)
    auc = roc_auc_score(yk, proba)
    acc = accuracy_score(yk, pred)
    print(f"  迁移测试集 AUC = {auc:.3f}, 准确率 = {acc:.3f}")

    # 解释
    if auc <= 0.55:
        print("  结论：跨数据集迁移能力很弱（AUC 接近 0.5），说明数据异构、")
        print("        结论不可直接搬移，必须依赖多源互补而非单一迁移。")
    else:
        print("  结论：存在一定迁移能力，说明年龄/压力两个共有特征具备跨源泛化性。")

    # 迁移 ROC 曲线
    fpr, tpr, _ = roc_curve(yk, proba)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"Transfer LR (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Cross-dataset Transfer ROC (Mendeley -> Kaggle)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_transfer_roc.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    datasets = load_datasets()
    print(f"清洗后规模: Mendeley={len(datasets['Mendeley'])}, Kaggle={len(datasets['Kaggle'])}, "
          f"UCI={len(datasets['UCI'])}, Luke={len(datasets['Luke'])}")

    # 任务1：单因素一致性对比
    table = consistency_analysis(datasets)
    print("\n" + "=" * 70)
    print("单因素结论一致性对比表")
    print("=" * 70)
    show = table[["factor", "dataset", "p", "label", "detail"]].copy()
    show["p"] = show["p"].map(lambda v: f"{v:.4f}")
    print(show.to_string(index=False))

    table.to_csv(REPORT_DIR / "05_consistency_table.csv", index=False)
    plot_consistency_heatmap(table)

    # 任务4：Cohen's Kappa
    compute_kappa(table)

    # 任务3：跨数据集迁移
    transfer_experiment(datasets["Mendeley"], datasets["Kaggle"])

    print("\n图表已保存到:", FIG_DIR)
    print("一致性表已保存到:", REPORT_DIR / "05_consistency_table.csv")


if __name__ == "__main__":
    main()
