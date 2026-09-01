# -*- coding: utf-8 -*-
"""
02_univariate_analysis.py —— 模块2：单因素分析（谁最容易脱发？）
================================================================
对 4 个多源数据集逐因素做"可视化 → 统计检验 → 一句话结论"：
    1. 年龄 vs 脱发率（Kaggle、UCI，t 检验）
    2. 性别差异（Mendeley、UCI，卡方）
    3. 遗传/家族史（Kaggle、Mendeley，卡方）
    4. 压力水平（Kaggle、Mendeley、Luke，卡方）
    5. 吸烟习惯（仅 Kaggle，卡方）
    6. Mendeley 独有因素（熬夜/水质/护发/贫血/睡眠，卡方）

约定：图表内文字统一英文，中文只出现在控制台结论中。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# ----------------------------------------------------------------------------
# 路径与全局配置
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
FIG_DIR = BASE_DIR / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")

# 有序类别顺序
STRESS_ORDER_KAGGLE = ["Low", "Moderate", "High"]
STRESS_ORDER_LUKE = ["Low", "Medium", "High", "Very High"]
HAIR_LOSS_LUKE = ["Few", "Medium", "Many", "A lot"]

# Mendeley 独有因素：列名 -> 展示名
MENDELEY_FACTORS = {
    "do_you_stay_up_late_at_night": "Stay Up Late",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems": "Water Quality",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair": "Hair Products/Chemicals",
    "do_you_have_anemia": "Anemia",
    "do_you_have_any_type_of_sleep_disturbance": "Sleep Disturbance",
}


# ----------------------------------------------------------------------------
# 数据读取与清洗
# ----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """列名统一：小写、空格/特殊字符转下划线。"""
    def _norm(name: str) -> str:
        s = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower())
        return re.sub(r"_+", "_", s).strip("_")
    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def clean_mendeley(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 Mendeley 问卷：修正错别字、剔除年龄异常值。"""
    df = df.copy()
    # 错别字：Yea -> Yes；\No -> No
    df = df.replace({"Yea": "Yes", "\\No": "No"})
    # 剔除年龄异常值（>100 视为录入错误，仅 218 一条）
    df = df[df["what_is_your_age"] <= 100].copy()
    return df


def clean_uci(df: pd.DataFrame) -> pd.DataFrame:
    """清洗 UCI：去除重复行。"""
    return df.drop_duplicates().copy()


def load_data() -> dict:
    """加载并清洗 4 个数据集，返回 {名称: DataFrame}。"""
    files = {
        "kaggle": "kaggle_hair_health.csv",
        "mendeley": "mendeley_hair_loss_survey.csv",
        "luke": "luke_hair_loss.csv",
        "uci": "uci_diabetes.csv",
    }
    data = {}
    for name, f in files.items():
        df = normalize_columns(pd.read_csv(DATA_DIR / f))
        if name == "mendeley":
            df = clean_mendeley(df)
        elif name == "uci":
            df = clean_uci(df)
        data[name] = df
    return data


# ----------------------------------------------------------------------------
# 统计与绘图工具
# ----------------------------------------------------------------------------
def binary_rate(df: pd.DataFrame, group_col: str, target_col: str,
                positive=("Yes", "1", "1.0")) -> pd.Series:
    """计算每个分组中"脱发"的比例（%）。"""
    s = df[target_col].astype(str).str.strip()
    pos = s.isin(positive)
    return pos.groupby(df[group_col]).mean() * 100


def reorder(rate: pd.Series, order: list) -> pd.Series:
    """按指定顺序重排 rate 序列（丢弃不存在的类别）。"""
    return rate.reindex([o for o in order if o in rate.index])


def chi2_test(df: pd.DataFrame, col_a: str, col_b: str):
    """卡方独立性检验，返回 (chi2 统计量, p 值, 列联表)。"""
    ct = pd.crosstab(df[col_a], df[col_b])
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return chi2, p, ct


def fmt_p(p: float) -> str:
    """格式化 p 值，便于输出。"""
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.3f}"


def rate_bar(ax, rate: pd.Series, title: str, color: str = "#4C72B0"):
    """在指定子图绘制"脱发率"柱状图并标注数值。"""
    rate = rate.dropna()
    bars = ax.bar(rate.index.astype(str), rate.values, color=color, width=0.6)
    for b, v in zip(bars, rate.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9)
    ax.set_title(title)
    ax.set_xlabel("Group")
    ax.set_ylabel("Hair Loss Rate (%)")
    top = rate.max() if len(rate) else 100
    ax.set_ylim(0, top * 1.25)


# ----------------------------------------------------------------------------
# 因素 1：年龄 vs 脱发率（Kaggle、UCI）
# ----------------------------------------------------------------------------
def analyze_age(k: pd.DataFrame, u: pd.DataFrame) -> None:
    """年龄与脱发的关系：箱线图 + 直方图 + t 检验。"""
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))

    # Kaggle：hair_loss 0/1 -> No/Yes
    kp = k.copy()
    kp["hair_loss_label"] = kp["hair_loss"].map({0: "No", 1: "Yes"})

    sns.boxplot(data=kp, x="hair_loss_label", y="age", order=["No", "Yes"], ax=axes[0, 0])
    axes[0, 0].set_title("Kaggle: Age by Hair Loss (Boxplot)")
    sns.histplot(data=kp, x="age", hue="hair_loss_label", ax=axes[0, 1], kde=True)
    axes[0, 1].set_title("Kaggle: Age Distribution by Hair Loss")

    sns.boxplot(data=u, x="alopecia", y="age", order=["No", "Yes"], ax=axes[1, 0])
    axes[1, 0].set_title("UCI: Age by Alopecia (Boxplot)")
    sns.histplot(data=u, x="age", hue="alopecia", ax=axes[1, 1], kde=True)
    axes[1, 1].set_title("UCI: Age Distribution by Alopecia")

    fig.suptitle("Age vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_age_vs_hairloss.png", dpi=300)
    plt.close(fig)

    # t 检验
    kg0 = k.loc[k["hair_loss"] == 0, "age"].dropna()
    kg1 = k.loc[k["hair_loss"] == 1, "age"].dropna()
    tk, pk = stats.ttest_ind(kg0, kg1)
    u_no = u.loc[u["alopecia"] == "No", "age"].dropna()
    u_yes = u.loc[u["alopecia"] == "Yes", "age"].dropna()
    tu, pu = stats.ttest_ind(u_no, u_yes)

    print("【年龄 vs 脱发】")
    print(f"  Kaggle: 不脱发均值 {kg0.mean():.1f} 岁 vs 脱发均值 {kg1.mean():.1f} 岁 (t 检验 {fmt_p(pk)})")
    print(f"  UCI:    无脱发均值 {u_no.mean():.1f} 岁 vs 有脱发均值 {u_yes.mean():.1f} 岁 (t 检验 {fmt_p(pu)})")


# ----------------------------------------------------------------------------
# 因素 2：性别差异（Mendeley、UCI）
# ----------------------------------------------------------------------------
def analyze_gender(m: pd.DataFrame, u: pd.DataFrame) -> None:
    """性别与脱发率：分组柱状图 + 卡方检验。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    r_m = binary_rate(m, "what_is_your_gender", "do_you_have_hair_fall_problem")
    rate_bar(axes[0], r_m, "Mendeley: Hair Loss Rate by Gender", "#DD8452")

    r_u = binary_rate(u, "gender", "alopecia")
    rate_bar(axes[1], r_u, "UCI: Alopecia Rate by Gender", "#55A868")

    fig.suptitle("Gender vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_gender_vs_hairloss.png", dpi=300)
    plt.close(fig)

    chi_m, p_m, _ = chi2_test(m, "what_is_your_gender", "do_you_have_hair_fall_problem")
    chi_u, p_u, _ = chi2_test(u, "gender", "alopecia")

    print("【性别 vs 脱发】")
    print(f"  Mendeley: 男 {r_m.get('Male', 0):.1f}% vs 女 {r_m.get('Female', 0):.1f}% (卡方 {fmt_p(p_m)})")
    print(f"  UCI:      男 {r_u.get('Male', 0):.1f}% vs 女 {r_u.get('Female', 0):.1f}% (卡方 {fmt_p(p_u)})")


# ----------------------------------------------------------------------------
# 因素 3：遗传/家族史（Kaggle、Mendeley）
# ----------------------------------------------------------------------------
def analyze_genetics(k: pd.DataFrame, m: pd.DataFrame) -> None:
    """遗传/家族史与脱发率：分组柱状图 + 卡方检验。"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    r_k = binary_rate(k, "genetics", "hair_loss")
    rate_bar(axes[0], r_k, "Kaggle: Hair Loss Rate by Genetics")

    r_m = binary_rate(m, "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
                      "do_you_have_hair_fall_problem")
    rate_bar(axes[1], r_m, "Mendeley: Hair Loss Rate by Family History", "#DD8452")

    fig.suptitle("Genetics / Family History vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_genetics_vs_hairloss.png", dpi=300)
    plt.close(fig)

    chi_k, p_k, _ = chi2_test(k, "genetics", "hair_loss")
    chi_m, p_m, _ = chi2_test(m, "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
                              "do_you_have_hair_fall_problem")

    print("【遗传/家族史 vs 脱发】")
    print(f"  Kaggle:   有遗传 {r_k.get('Yes', 0):.1f}% vs 无遗传 {r_k.get('No', 0):.1f}% (卡方 {fmt_p(p_k)})")
    print(f"  Mendeley: 有家族史 {r_m.get('Yes', 0):.1f}% vs 无家族史 {r_m.get('No', 0):.1f}% (卡方 {fmt_p(p_m)})")


# ----------------------------------------------------------------------------
# 因素 4：压力水平（Kaggle、Mendeley、Luke）
# ----------------------------------------------------------------------------
def analyze_stress(k: pd.DataFrame, m: pd.DataFrame, l: pd.DataFrame) -> None:
    """压力与脱发率：三数据集分组柱状图 + 卡方检验。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    r_k = reorder(binary_rate(k, "stress", "hair_loss"), STRESS_ORDER_KAGGLE)
    rate_bar(axes[0], r_k, "Kaggle: Hair Loss Rate by Stress")

    r_m = binary_rate(m, "do_you_have_too_much_stress", "do_you_have_hair_fall_problem")
    rate_bar(axes[1], r_m, "Mendeley: Hair Loss Rate by Stress", "#DD8452")

    # Luke：hair_loss 二值化（Many/A lot 视为严重脱发）
    r_l = reorder(binary_rate(l, "stress_level", "hair_loss", positive=("Many", "A lot")),
                  STRESS_ORDER_LUKE)
    rate_bar(axes[2], r_l, "Luke: Severe Hair Loss Rate by Stress", "#55A868")

    fig.suptitle("Stress vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_stress_vs_hairloss.png", dpi=300)
    plt.close(fig)

    _, p_k, _ = chi2_test(k, "stress", "hair_loss")
    _, p_m, _ = chi2_test(m, "do_you_have_too_much_stress", "do_you_have_hair_fall_problem")
    # Luke 需先二值化
    l2 = l.copy()
    l2["severe"] = l2["hair_loss"].isin(["Many", "A lot"])
    _, p_l, _ = chi2_test(l2, "stress_level", "severe")

    print("【压力 vs 脱发】")
    print(f"  Kaggle:   脱发率随压力 {dict(r_k)} (卡方 {fmt_p(p_k)})")
    print(f"  Mendeley: 高压 {r_m.get('Yes', 0):.1f}% vs 低压 {r_m.get('No', 0):.1f}% (卡方 {fmt_p(p_m)})")
    print(f"  Luke:     严重脱发率随压力 {dict(r_l)} (卡方 {fmt_p(p_l)})")


# ----------------------------------------------------------------------------
# 因素 5：吸烟习惯（仅 Kaggle）
# ----------------------------------------------------------------------------
def analyze_smoking(k: pd.DataFrame) -> None:
    """吸烟与脱发率：对比柱状图 + 卡方检验。"""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    r = binary_rate(k, "smoking", "hair_loss")
    rate_bar(ax, r, "Kaggle: Hair Loss Rate by Smoking")
    fig.suptitle("Smoking vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_smoking_vs_hairloss.png", dpi=300)
    plt.close(fig)

    _, p, _ = chi2_test(k, "smoking", "hair_loss")
    print("【吸烟 vs 脱发】")
    print(f"  Kaggle: 吸烟 {r.get('Yes', 0):.1f}% vs 不吸烟 {r.get('No', 0):.1f}% (卡方 {fmt_p(p)})")


# ----------------------------------------------------------------------------
# 因素 6：Mendeley 独有因素（熬夜/水质/护发/贫血/睡眠）
# ----------------------------------------------------------------------------
def analyze_mendeley_factors(m: pd.DataFrame) -> None:
    """Mendeley 生活习惯类独有因素：分组柱状图 + 卡方检验。"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.ravel()

    print("【Mendeley 独有因素 vs 脱发】")
    for i, (col, label) in enumerate(MENDELEY_FACTORS.items()):
        r = binary_rate(m, col, "do_you_have_hair_fall_problem")
        rate_bar(axes[i], r, label, color="#DD8452")
        _, p, _ = chi2_test(m, col, "do_you_have_hair_fall_problem")
        print(f"  {label}: Yes {r.get('Yes', 0):.1f}% vs No {r.get('No', 0):.1f}% (卡方 {fmt_p(p)})")

    axes[5].axis("off")  # 隐藏多余子图
    fig.suptitle("Mendeley Lifestyle Factors vs Hair Loss", fontsize=15)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_mendeley_lifestyle_factors.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    data = load_data()
    k, m, l, u = data["kaggle"], data["mendeley"], data["luke"], data["uci"]

    print("=" * 70)
    print(f"数据规模(清洗后): Kaggle={len(k)}, Mendeley={len(m)}, Luke={len(l)}, UCI={len(u)}")
    print("=" * 70)

    analyze_age(k, u)
    analyze_gender(m, u)
    analyze_genetics(k, m)
    analyze_stress(k, m, l)
    analyze_smoking(k)
    analyze_mendeley_factors(m)

    print("\n图表已保存到:", FIG_DIR)


if __name__ == "__main__":
    main()
