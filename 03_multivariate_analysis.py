# -*- coding: utf-8 -*-
"""
03_multivariate_analysis.py —— 模块3：多因素关联分析
====================================================
核心问题：因素之间如何相互作用？

对 Mendeley 问卷数据，从「单因素」上升到「多因素交互」：
    1. 特征关联矩阵（Cramér's V）
    2. 分层交互分析（stratification，效应修饰）
    3. 交互显著性检验（逻辑回归 Logit 交互项 p 值）
    4. 高危组合识别（因素组合脱发率排序）

约定：图表文字统一英文，中文仅用于控制台结论与结论文档。
"""
import re
import warnings
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

import statsmodels.api as sm

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------
# 路径与全局配置
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "reports" / "figures"
REPORT_DIR = BASE_DIR / "reports"
FIG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

TARGET = "do_you_have_hair_fall_problem"
AGE_COL = "what_is_your_age"
GENDER_COL = "what_is_your_gender"
YES_NO_COLS = [
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_stay_up_late_at_night",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair",
    "do_you_have_anemia",
    "do_you_have_too_much_stress",
]

# 图表用英文短名
DISPLAY = {
    AGE_COL: "Age",
    GENDER_COL: "Gender",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue": "Family History",
    "do_you_stay_up_late_at_night": "Stay Up Late",
    "do_you_have_any_type_of_sleep_disturbance": "Sleep Disturbance",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems": "Water Quality",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair": "Hair Products",
    "do_you_have_anemia": "Anemia",
    "do_you_have_too_much_stress": "Stress",
    TARGET: "Hair Loss",
}

# 高危组合枚举用 Top5 特征（M4 重要性 Top）
TOP5 = [
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_have_too_much_stress",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_have_anemia",
]


# ----------------------------------------------------------------------------
# 数据读取与清洗（复用模块4 逻辑）
# ----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一：小写、空格/特殊字符转下划线。"""
    def _norm(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return re.sub(r"_+", "_", s).strip("_")
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def load_and_prepare() -> pd.DataFrame:
    """加载、清洗并编码，返回编码后的 DataFrame（含 age_group 分箱列）。"""
    df = normalize_columns(pd.read_csv(DATA_DIR / "mendeley_hair_loss_survey.csv"))
    # 清洗：错别字、年龄异常值
    df = df.replace({"Yea": "Yes", "\\No": "No"})
    df = df[df[AGE_COL] <= 100].copy()
    df = df[[AGE_COL, GENDER_COL, TARGET] + YES_NO_COLS].copy()

    # 目标与 Yes/No 特征映射
    df[TARGET] = df[TARGET].map({"No": 0, "Yes": 1})
    for col in YES_NO_COLS:
        df[col] = df[col].map({"No": 0, "Yes": 1})
    df[GENDER_COL] = df[GENDER_COL].map({"Female": 0, "Male": 1})
    df[AGE_COL] = df[AGE_COL].astype(float)

    # 年龄分箱（用于关联矩阵）
    bins = [0, 20, 25, 30, 120]
    labels = ["<20", "20-24", "25-29", "30+"]
    df["age_group"] = pd.cut(df[AGE_COL], bins=bins, labels=labels, right=False)

    return df


# ----------------------------------------------------------------------------
# 1. 特征关联矩阵（Cramér's V）
# ----------------------------------------------------------------------------
def cramers_v(a, b) -> float:
    """两个分类变量间的 Cramér's V（0~1，越大关联越强）。"""
    ct = pd.crosstab(a, b)
    n = ct.values.sum()
    if n == 0:
        return 0.0
    chi2 = chi2_contingency(ct.values, correction=False)[0]
    min_dim = min(ct.shape) - 1
    if min_dim == 0:
        return 0.0
    return float(np.sqrt(chi2 / (n * min_dim)))


def association_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """计算 9 特征（年龄分箱）+ 目标的 Cramér's V 关联矩阵。"""
    vars_ = YES_NO_COLS + [GENDER_COL, "age_group", TARGET]
    labels = [DISPLAY[c] if c in DISPLAY else "Age Group" for c in vars_]
    n = len(vars_)
    mat = pd.DataFrame(np.eye(n), index=labels, columns=labels)
    for i, j in combinations(range(n), 2):
        v = cramers_v(df[vars_[i]], df[vars_[j]])
        mat.iloc[i, j] = v
        mat.iloc[j, i] = v
    return mat


def plot_association_heatmap(mat: pd.DataFrame) -> None:
    """Cramér's V 关联热力图。"""
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(mat, annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, linewidths=0.5, square=True, ax=ax)
    ax.set_title("Cramer's V Association Matrix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_association_heatmap.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 2. 交互显著性检验（逻辑回归交互项）
# ----------------------------------------------------------------------------
def interaction_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    """对 7 个 Yes/No 特征两两建立 Logit 交互模型，输出交互项 p 值。"""
    y = df[TARGET].values
    rows = []
    for a, b in combinations(YES_NO_COLS, 2):
        try:
            X = df[[a, b]].copy()
            X["interaction"] = X[a] * X[b]
            X = sm.add_constant(X)
            model = sm.Logit(y, X).fit(disp=0)
            coef = model.params["interaction"]
            p = model.pvalues["interaction"]
        except Exception:
            coef, p = np.nan, np.nan
        rows.append({
            "feature_a": DISPLAY[a],
            "feature_b": DISPLAY[b],
            "coef": coef,
            "p_value": p,
            "significant": p < 0.05 if pd.notna(p) else False,
        })
    res = pd.DataFrame(rows).sort_values("p_value").reset_index(drop=True)
    return res


# ----------------------------------------------------------------------------
# 3. 分层交互分析（stratification）
# ----------------------------------------------------------------------------
def stratified_rates(df: pd.DataFrame, a, b) -> pd.DataFrame:
    """控制 A 后，B 的脱发率；返回 4 个组合的 (n, 脱发率)。"""
    rows = []
    for a_val, b_val in product([0, 1], [0, 1]):
        sub = df[(df[a] == a_val) & (df[b] == b_val)]
        n = len(sub)
        rate = sub[TARGET].mean() if n > 0 else np.nan
        rows.append({
            "A": a_val, "B": b_val,
            "n": n,
            "hair_loss_rate": rate,
        })
    return pd.DataFrame(rows)


def plot_interaction_bars(df: pd.DataFrame, pairs) -> None:
    """对交互显著的 2~3 对，画分组堆叠柱状图（组合脱发/不脱发比例）。"""
    n_pairs = len(pairs)
    fig, axes = plt.subplots(1, n_pairs, figsize=(6 * n_pairs, 5))
    if n_pairs == 1:
        axes = [axes]

    for ax, (a, b, _) in zip(axes, pairs):
        cats = [f"{DISPLAY[a]}=No\n{DISPLAY[b]}=No",
                f"{DISPLAY[a]}=No\n{DISPLAY[b]}=Yes",
                f"{DISPLAY[a]}=Yes\n{DISPLAY[b]}=No",
                f"{DISPLAY[a]}=Yes\n{DISPLAY[b]}=Yes"]
        no_pct, yes_pct = [], []
        for a_val, b_val in product([0, 1], [0, 1]):
            sub = df[(df[a] == a_val) & (df[b] == b_val)]
            n = len(sub)
            if n == 0:
                no_pct.append(0); yes_pct.append(0)
            else:
                yes_pct.append(sub[TARGET].mean() * 100)
                no_pct.append(100 - yes_pct[-1])
        ax.bar(cats, no_pct, label="No Hair Loss", color="#4C72B0")
        ax.bar(cats, yes_pct, bottom=no_pct, label="Hair Loss", color="#C44E52")
        for i, (n0, y0) in enumerate(zip(no_pct, yes_pct)):
            ax.text(i, n0 + y0 / 2, f"{y0:.0f}%", ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold")
        ax.set_ylabel("Percentage (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"{DISPLAY[a]} x {DISPLAY[b]}")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", labelsize=8)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_interaction_bar.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. 高危组合识别
# ----------------------------------------------------------------------------
def combination_risk(df: pd.DataFrame, min_n: int = 10) -> pd.DataFrame:
    """枚举 Top5 特征组合，计算脱发率，返回样本量足够的组合按脱发率降序。"""
    rows = []
    for combo in product([0, 1], repeat=len(TOP5)):
        sub = df
        label_parts = []
        for col, val in zip(TOP5, combo):
            sub = sub[sub[col] == val]
            if val == 1:
                label_parts.append(DISPLAY[col])
        n = len(sub)
        if n < min_n:
            continue
        rate = sub[TARGET].mean()
        label = " + ".join(label_parts) if label_parts else "None (all No)"
        rows.append({"combination": label, "n": n, "hair_loss_rate": rate})
    res = pd.DataFrame(rows).sort_values("hair_loss_rate", ascending=False)
    return res.reset_index(drop=True)


def plot_combination_risk(combo_df: pd.DataFrame) -> None:
    """高危组合风险条形图（Top 组合）。"""
    top = combo_df.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(top["combination"], top["hair_loss_rate"] * 100, color="#C44E52")
    ax.set_xlabel("Hair Loss Rate (%)")
    ax.set_title("Top Risk Combinations (hair loss rate)")
    for b, (_, row) in zip(bars, top.iterrows()):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{row['hair_loss_rate'] * 100:.0f}% (n={row['n']})",
                va="center", fontsize=8)
    ax.set_xlim(0, 105)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_combination_risk.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 结论文档
# ----------------------------------------------------------------------------
def write_report(df, assoc_mat, inter_df, combo_df, n_samples) -> None:
    """生成中文结论文档。"""
    overall_rate = df[TARGET].mean() * 100
    lines = [
        "# 模块3：多因素关联分析结论文档",
        "",
        f"- 数据：Mendeley 问卷（清洗后 {n_samples} 条，总体脱发率 {overall_rate:.1f}%）",
        f"- 目标变量：{DISPLAY[TARGET]}（0/1）",
        "",
        "## 1. 特征关联矩阵",
        "对 9 个特征（年龄分箱）+ 目标计算 Cramér's V。与脱发关联最强的因素：",
        "",
    ]
    # 与目标关联强度排序
    target_row = assoc_mat.loc[DISPLAY[TARGET]].drop(DISPLAY[TARGET]).sort_values(ascending=False)
    for name, v in target_row.items():
        lines.append(f"- {name}: Cramér's V = {v:.3f}")
    lines.append("")

    lines += [
        "## 2. 交互显著性（逻辑回归交互项）",
        "7 个 Yes/No 特征两两建立 Logit 交互模型，交互项 p 值最小的组合如下（p<0.05 视为交互显著）：",
        "",
        "| 因素 A | 因素 B | 交互项系数 | p 值 | 是否显著 |",
        "|---|---|---|---|---|",
    ]
    for _, row in inter_df.head(10).iterrows():
        lines.append(
            f"| {row['feature_a']} | {row['feature_b']} | {row['coef']:.3f} | "
            f"{row['p_value']:.4f} | {'显著' if row['significant'] else '不显著'} |")
    lines.append("")

    lines += [
        "## 3. 高危组合",
        "Top5 特征组合中脱发率最高的组合（仅保留样本量足够的组合）：",
        "",
        "| 组合 | 样本量 | 脱发率 |",
        "|---|---|---|",
    ]
    for _, row in combo_df.head(10).iterrows():
        lines.append(f"| {row['combination']} | {int(row['n'])} | {row['hair_loss_rate'] * 100:.1f}% |")
    lines.append("")

    lines += [
        "## 4. 结论",
        "（此段在脚本运行后补充关键发现，见控制台输出。）",
        "",
    ]
    (REPORT_DIR / "03_multivariate_findings.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结论文档已生成: {REPORT_DIR / '03_multivariate_findings.md'}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    df = load_and_prepare()
    n_samples = len(df)
    print(f"数据规模: {n_samples} 条, 脱发率 {df[TARGET].mean() * 100:.1f}%")

    # 1. 关联矩阵
    assoc_mat = association_matrix(df)
    plot_association_heatmap(assoc_mat)
    print("\n" + "=" * 70)
    print("1. 与脱发关联最强的因素（Cramér's V）")
    print("=" * 70)
    target_row = assoc_mat.loc[DISPLAY[TARGET]].drop(DISPLAY[TARGET]).sort_values(ascending=False)
    for name, v in target_row.items():
        print(f"  {name}: {v:.3f}")

    # 2. 交互显著性
    inter_df = interaction_pvalues(df)
    print("\n" + "=" * 70)
    print("2. 两两交互项 p 值（Logit，Top 10）")
    print("=" * 70)
    print(inter_df.head(10).to_string(index=False))

    # 3. 分层分析 + 交互图（取交互最显著的 2 对）
    sig_pairs = inter_df.head(2)
    pairs = []
    for _, row in sig_pairs.iterrows():
        a = YES_NO_COLS[[DISPLAY[c] for c in YES_NO_COLS].index(row["feature_a"])]
        b = YES_NO_COLS[[DISPLAY[c] for c in YES_NO_COLS].index(row["feature_b"])]
        pairs.append((a, b, row["p_value"]))
    print("\n" + "=" * 70)
    print("3. 分层交互分析（效应修饰）")
    print("=" * 70)
    for a, b, p in pairs:
        strat = stratified_rates(df, a, b)
        print(f"\n{DISPLAY[a]} x {DISPLAY[b]}  (交互 p={p:.4f})")
        print(strat.to_string(index=False))
    if pairs:
        plot_interaction_bars(df, pairs)

    # 4. 高危组合
    combo_df = combination_risk(df, min_n=10)
    print("\n" + "=" * 70)
    print("4. 高危组合（Top 10）")
    print("=" * 70)
    print(combo_df.head(10).to_string(index=False))
    plot_combination_risk(combo_df)

    # 结论文档
    write_report(df, assoc_mat, inter_df, combo_df, n_samples)
    print(f"\n图表已保存到: {FIG_DIR}")


if __name__ == "__main__":
    main()
