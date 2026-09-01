# -*- coding: utf-8 -*-
"""
01_data_profile.py —— 模块1：数据画像与质量评估
====================================================
功能：
    1. 加载 4 个多源数据集，输出基础画像（形状/类型/样本/缺失/重复/描述统计/唯一值）
    2. 生成 4 张多源对比可视化图表（保存到 reports/figures/）
    3. 自动生成数据质量评估报告（保存到 reports/data_quality_report.md）

数据集与目标变量说明：
    - Kaggle   -> kaggle_hair_health.csv       目标: hair_loss (0/1, 映射为 No/Yes)
    - Mendeley -> mendeley_hair_loss_survey.csv 目标: do_you_have_hair_fall_problem (Yes/No)
    - Luke     -> luke_hair_loss.csv            目标: hair_loss (Few/Medium/Many/A lot, 有序)
    - UCI      -> uci_diabetes.csv              目标: alopecia (Yes/No, 若为 1/2 则自动映射)

注：UCI 的 Alopecia 原始编码本文件已为 Yes/No，脚本兼容 1=Yes, 2=No 的编码情况。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------------------------------------------
# 全局配置
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 中文字体支持
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-whitegrid")

# 数据集元信息：逻辑名 -> (文件名, 展示名, 来源说明, 目标变量列)
DATASETS = {
    "Kaggle": {
        "file": "kaggle_hair_health.csv",
        "label": "Kaggle Hair Health",
        "source": "结构化医学因素（横截面）",
        "target": "hair_loss",
    },
    "Mendeley": {
        "file": "mendeley_hair_loss_survey.csv",
        "label": "Mendeley 脱发问卷",
        "source": "在线问卷（横截面）",
        "target": "do_you_have_hair_fall_problem",
    },
    "Luke": {
        "file": "luke_hair_loss.csv",
        "label": "Luke 脱发日记",
        "source": "个人纵向记录（时间序列）",
        "target": "hair_loss",
    },
    "UCI": {
        "file": "uci_diabetes.csv",
        "label": "UCI 糖尿病风险",
        "source": "医院/临床问卷（横截面）",
        "target": "alopecia",
    },
}

# 目标变量展示配置
TARGET_CONFIG = {
    "Kaggle": {
        "title": "Baldness (No/Yes)",
        "map": {"0": "No", "1": "Yes", "0.0": "No", "1.0": "Yes"},
    },
    "Mendeley": {"title": "Hair Fall (Self-reported)"},
    "Luke": {
        "title": "Daily Hair Loss (Ordinal)",
        "order": ["Few", "Medium", "Many", "A lot"],
    },
    "UCI": {
        "title": "Alopecia (Symptom)",
        # 兼容 1=Yes, 2=No 以及已映射的 Yes/No
        "map": {"1": "Yes", "2": "No", "Yes": "Yes", "No": "No"},
    },
}


# ----------------------------------------------------------------------------
# 工具函数
# ----------------------------------------------------------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """将列名统一为小写、去空格、特殊字符替换为下划线，便于跨数据集对比。"""
    def _norm(name: str) -> str:
        s = str(name).strip().lower()
        s = re.sub(r"[^a-z0-9]+", "_", s)   # 非字母数字 -> 下划线
        s = re.sub(r"_+", "_", s).strip("_")  # 合并并去除首尾下划线
        return s

    df = df.copy()
    df.columns = [_norm(c) for c in df.columns]
    return df


def load_dataset(name: str) -> pd.DataFrame | None:
    """加载单个数据集，文件不存在时给出友好提示并返回 None。"""
    meta = DATASETS[name]
    path = DATA_DIR / meta["file"]
    if not path.exists():
        print(f"[警告] 文件不存在，已跳过 {name}: {path}")
        return None
    df = pd.read_csv(path)
    df = normalize_columns(df)
    return df


def missing_rate(series: pd.Series) -> float:
    """计算单个字段的缺失比例（百分比）。"""
    return series.isna().mean() * 100


# ----------------------------------------------------------------------------
# 1. 基础画像（控制台输出）
# ----------------------------------------------------------------------------
def print_basic_profile(name: str, df: pd.DataFrame) -> None:
    """打印单个数据集的基础画像信息。"""
    print("\n" + "=" * 90)
    print(f"数据集: {name}  ({DATASETS[name]['label']})")
    print("=" * 90)
    print(f"形状(shape): {df.shape}")
    print(f"\n列名与数据类型(dtypes):\n{df.dtypes}")
    print(f"\n前 5 行(head):\n{df.head().to_string()}")
    print(f"\n缺失值数量: {int(df.isna().sum().sum())}  缺失比例: "
          f"{missing_rate(pd.Series(df.isna().sum().sum(), index=[0]).repeat(1)).round(2)} % "
          f"(整体)")
    print(f"重复行数量: {int(df.duplicated().sum())}")
    numeric_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(exclude=np.number).columns
    if len(numeric_cols):
        print(f"\n数值型字段描述统计(describe):\n{df[numeric_cols].describe().to_string()}")
    if len(cat_cols):
        print(f"\n分类型字段唯一值数量(nunique):\n{df[cat_cols].nunique().to_string()}")


# ----------------------------------------------------------------------------
# 2. 可视化
# ----------------------------------------------------------------------------
def plot_dataset_size(frames: dict) -> None:
    """图表A：数据集规模对比柱状图。"""
    sizes = pd.Series({k: len(v) for k, v in frames.items()})
    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = sns.color_palette("deep", len(sizes))
    bars = ax.bar(sizes.index, sizes.values, color=colors, width=0.6)
    for b, v in zip(bars, sizes.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 8, f"{v:,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Record Count")
    ax.set_title("Multi-source Dataset Size Comparison")
    ax.set_ylim(0, sizes.max() * 1.18)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_dataset_size_comparison.png", dpi=300)
    plt.close(fig)


def plot_field_coverage(frames: dict) -> None:
    """图表B：字段覆盖度矩阵热力图（行=字段，列=数据集，1=覆盖 0=未覆盖）。"""
    all_cols = sorted({c for df in frames.values() for c in df.columns})
    coverage = pd.DataFrame(
        0, index=all_cols, columns=list(frames.keys()), dtype=int
    )
    for dname, df in frames.items():
        for col in set(df.columns):
            coverage.loc[col, dname] = 1

    height = max(6, len(all_cols) * 0.28)
    fig, ax = plt.subplots(figsize=(7, height))
    cmap = sns.color_palette(["#f0f0f0", "#4C72B0"], as_cmap=True)
    sns.heatmap(coverage, annot=True, fmt="d", cmap=cmap,
                cbar=False, linewidths=0.4, linecolor="white",
                annot_kws={"fontsize": 8}, ax=ax)
    ax.set_title("Multi-source Field Coverage Matrix")
    ax.set_ylabel("Field")
    ax.set_xlabel("Dataset")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_field_coverage_matrix.png", dpi=300)
    plt.close(fig)


def plot_missing_values(frames: dict) -> None:
    """图表C：各数据集字段缺失值比例水平条形图。"""
    rows = []
    for dname, df in frames.items():
        for col in df.columns:
            mr = missing_rate(df[col])
            if mr > 0:
                rows.append({"field": f"{dname} · {col}", "rate": mr, "dataset": dname})
    if not rows:
        # 无缺失时生成一张占位说明图
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No missing values in any dataset", ha="center", va="center", fontsize=14)
        ax.axis("off")
        ax.set_title("Missing Value Distribution by Dataset")
        fig.savefig(FIGURES_DIR / "01_missing_values_comparison.png", dpi=300)
        plt.close(fig)
        return

    miss = pd.DataFrame(rows).sort_values("rate", ascending=True)
    fig, ax = plt.subplots(figsize=(9, max(4, len(miss) * 0.5)))
    palette = {"Kaggle": "#4C72B0", "Mendeley": "#DD8452",
               "Luke": "#55A868", "UCI": "#C44E52"}
    sns.barplot(data=miss, x="rate", y="field",
                hue="dataset", palette=palette, ax=ax, legend=False)
    for p in ax.patches:
        ax.text(p.get_width() + 0.3, p.get_y() + p.get_height() / 2,
                f"{p.get_width():.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Missing Rate (%)")
    ax.set_ylabel("Field")
    ax.set_title("Missing Value Distribution by Dataset")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_missing_values_comparison.png", dpi=300)
    plt.close(fig)


def _clean_target(name: str, df: pd.DataFrame) -> pd.Series:
    """提取并清洗目标变量，用于分布图。"""
    col = DATASETS[name]["target"]
    cfg = TARGET_CONFIG[name]
    series = df[col].astype(str).str.strip()
    if "map" in cfg:
        series = series.map(lambda x: cfg["map"].get(x, x))
    if "order" in cfg:
        series = pd.Categorical(series, categories=cfg["order"], ordered=True)
    return series


def plot_target_distribution(frames: dict) -> None:
    """图表D：目标变量分布对比（2x2 子图）。"""
    names = [k for k in frames.keys() if DATASETS[k]["target"] in frames[k].columns]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.ravel()
    for i, name in enumerate(names):
        ax = axes[i]
        series = _clean_target(name, frames[name])
        vc = series.value_counts()
        sns.barplot(x=vc.index.astype(str), y=vc.values, ax=ax,
                    color=sns.color_palette("deep", len(names))[i])
        for p, v in zip(ax.patches, vc.values):
            ax.text(p.get_x() + p.get_width() / 2, p.get_height() + 0.5,
                    str(int(v)), ha="center", va="bottom", fontsize=10)
        ax.set_title(f"{name} — {TARGET_CONFIG[name]['title']}")
        ax.set_xlabel("Category")
        ax.set_ylabel("Count")
    # 隐藏多余子图
    for j in range(len(names), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Target Variable Distribution Comparison", fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "01_target_distribution.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 3. 汇总表 + 报告生成
# ----------------------------------------------------------------------------
def build_summary_table(frames: dict) -> pd.DataFrame:
    """构建数据集基础信息汇总表。"""
    records = []
    for name, df in frames.items():
        target_col = DATASETS[name]["target"]
        target_desc = target_col if target_col in df.columns else "无目标变量"
        records.append({
            "数据集": f"{name}（{DATASETS[name]['label']}）",
            "记录数": len(df),
            "字段数": df.shape[1],
            "缺失值总数": int(df.isna().sum().sum()),
            "重复行数": int(df.duplicated().sum()),
            "目标变量": target_desc,
        })
    return pd.DataFrame(records)


def _md_table(df: pd.DataFrame) -> str:
    """将 DataFrame 转为 Markdown 表格字符串。"""
    header = "| " + " | ".join(df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(df.columns)) + " |"
    body = "\n".join(
        "| " + " | ".join(str(x) for x in row) + " |" for row in df.itertuples(index=False)
    )
    return "\n".join([header, sep, body])


def generate_report(frames: dict, summary: pd.DataFrame) -> None:
    """自动生成 Markdown 数据质量评估报告。"""
    missing_frames = [k for k, v in frames.items() if v.isna().sum().sum() > 0]
    dup_frames = [k for k, v in frames.items() if v.duplicated().sum() > 0]

    missing_note = "、".join(missing_frames) if missing_frames else "无"
    dup_note = "、".join(dup_frames) if dup_frames else "无"

    content = f"""# 数据质量评估报告

## 1. 项目概述

本项目面向"脱发影响因素分析与可视化"，采用 **4 个互补的公开多源数据集**，从
不同维度刻画脱发的潜在影响因素。各数据集在样本来源、字段维度与目标变量上互为补充，
单靠任何一个数据集都无法完整覆盖脱发的全部潜在因素，因此采用多源整合策略。

| 数据集 | 来源 | 特点 |
|---|---|---|
| Kaggle Hair Health | 结构化医学因素 | 字段最全（遗传/激素/疾病/营养/生活习惯），作为主数据集 |
| Mendeley 脱发问卷 | 在线问卷 | 补充"护发产品、水质、熬夜、贫血"等独特生活习惯维度 |
| Luke 脱发日记 | 个人纵向记录 | 时间序列，用于后续跨数据集模型验证 |
| UCI 糖尿病风险 | 临床问卷 | 含 Alopecia 字段，用于跨疾病关联分析 |

## 2. 数据集基础信息汇总

{_md_table(summary)}

## 3. 数据质量评估结论

- **缺失值**：存在缺失的数据集为 `{missing_note}`；其余数据集字段完整。
- **重复行**：存在重复行的数据集为 `{dup_note}`。
- **Kaggle 数据集**字段最全、无缺失，质量最高，适合作为主数据集建模。
- **Mendeley 数据集**补充了护发产品、水质、熬夜、贫血等独特生活习惯维度，是 Kaggle
  数据不具备的，能丰富分析视角。
- **Luke 数据集**为独立时间序列来源，字段维度不同，适合用于验证结论的普适性，而非直接合并。
- **UCI 数据集**通过 Alopecia 字段关联脱发与糖尿病，支持跨疾病分析。

## 4. 后续分析建议

1. **主数据集**：以 Kaggle Hair Health 为主，进行单因素与多因素建模。
2. **补充维度**：用 Mendeley 问卷补充生活习惯类因素的探索性分析。
3. **交叉验证**：用 Luke 时间序列数据验证"压力/熬夜等与脱发关系"结论的跨源一致性。
4. **跨疾病关联**：用 UCI 糖尿病数据对比"有/无脱发"人群的糖尿病患病率，识别共同风险因素。
5. 分析前需对部分数据做清洗（如错别字、异常值、重复行），再进入建模阶段。
"""
    (REPORTS_DIR / "data_quality_report.md").write_text(content, encoding="utf-8")
    print(f"\n报告已生成: {REPORTS_DIR / 'data_quality_report.md'}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    # 加载所有数据集
    frames = {}
    for name in DATASETS:
        df = load_dataset(name)
        if df is not None:
            frames[name] = df

    if not frames:
        print("未加载到任何数据集，请检查 data/raw/ 目录。")
        return

    # 1. 基础画像
    for name, df in frames.items():
        print_basic_profile(name, df)

    # 2. 可视化
    plot_dataset_size(frames)
    plot_field_coverage(frames)
    plot_missing_values(frames)
    plot_target_distribution(frames)
    print("\n图表已保存到:", FIGURES_DIR)

    # 3. 汇总 + 报告
    summary = build_summary_table(frames)
    print("\n" + "=" * 90)
    print("【数据集基础信息汇总表】")
    print(summary.to_string(index=False))
    generate_report(frames, summary)


if __name__ == "__main__":
    main()
