# -*- coding: utf-8 -*-
"""M6 跨疾病关联分析：脱发与糖尿病

产物: reports/m6_sankey.html, reports/m6_results.json
依赖: pandas, numpy, scipy, plotly
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

ROOT = Path(__file__).resolve().parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "reports"
OUT.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    df = pd.read_csv(RAW / name, skipinitialspace=True)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].astype(str).str.strip()
    return df


def yes(s: pd.Series) -> pd.Series:
    """兼容 Yes/Y/Yea 等脏值，统一识别"是"。"""
    return s.str.lower().str.startswith("y")


def binary_assoc(df: pd.DataFrame, col: str) -> dict:
    """二值因素 col 与糖尿病(class)的卡方 + OR（Yes 相对 No 的糖尿病优势比）。"""
    y = yes(df[col])
    t_pos = df["class"] == "Positive"
    a = int((y & t_pos).sum()); b = int((y & ~t_pos).sum())
    c = int((~y & t_pos).sum()); d = int((~y & ~t_pos).sum())
    chi2, p = stats.chi2_contingency(pd.crosstab(df[col], df["class"]))[:2]
    odds = (a * d) / (b * c) if b * c != 0 else None
    return {"factor": col,
            "odds_ratio_diabetes": round(float(odds), 2) if odds is not None else None,
            "chi2": round(float(chi2), 2), "p_value": float(p),
            "significant": bool(p < 0.05)}


# ---------- A. 真共现：UCI 自带 Alopecia × 糖尿病 ----------
uci = read_csv("uci_diabetes.csv").drop_duplicates().reset_index(drop=True)  # 去重：520→251
pos, neg = uci[uci["class"] == "Positive"], uci[uci["class"] == "Negative"]
p_pos, p_neg = yes(pos["Alopecia"]).mean(), yes(neg["Alopecia"]).mean()
a = int(yes(pos["Alopecia"]).sum()); b = len(pos) - a
c = int(yes(neg["Alopecia"]).sum()); d = len(neg) - c
chi2, pval = stats.chi2_contingency(pd.crosstab(uci["Alopecia"], uci["class"]))[:2]
odds = (a * d) / (b * c)
se = float(np.sqrt(1 / a + 1 / b + 1 / c + 1 / d))
ci = (odds * np.exp(-1.96 * se), odds * np.exp(1.96 * se))

# 背景脱发率（仅作对照）
kag, surv, luke = (read_csv(f) for f in
                   ["kaggle_hair_health.csv", "mendeley_hair_loss_survey.csv", "luke_hair_loss.csv"])
background = {
    "kaggle_hair_health": round(float(kag["Hair Loss"].mean()), 3),
    "mendeley_survey": round(float(yes(surv["Do you have hair fall problem ?"]).mean()), 3),
    "luke_severe": round(float(luke["hair_loss"].isin(["Many", "A lot"]).mean()), 3),
}

# ---------- B. 共同风险因素：脱发侧词典 + 糖尿病侧关联检验 ----------
N = {"kaggle_hair_health": len(kag), "luke_hair_loss": len(luke),
     "mendeley_survey": len(surv), "uci_diabetes": len(uci)}

# 脱发侧风险因素类别 → 证据数据源
HAIR_FACTORS = {
    "压力":         ["kaggle_hair_health", "luke_hair_loss", "mendeley_survey"],
    "熬夜与睡眠":   ["luke_hair_loss", "mendeley_survey"],
    "吸烟":         ["kaggle_hair_health"],
    "年龄":         ["kaggle_hair_health", "mendeley_survey"],
    "肥胖与体重":   ["kaggle_hair_health"],
    "遗传/家族史":  ["kaggle_hair_health", "mendeley_survey"],
    "慢性病史":     ["kaggle_hair_health", "mendeley_survey"],
    "营养缺乏/贫血": ["kaggle_hair_health", "mendeley_survey"],
    "激素/内分泌":  ["kaggle_hair_health"],
}

# 糖尿病侧：UCI 中可检验的列 → 所属类别（其余 UCI 症状无脱发侧对应，不参与"共同"判定）
UCI_CAT = {
    "Age": "年龄",
    "Obesity": "肥胖与体重",
    "sudden weight loss": "肥胖与体重",
}

# UCI 13 个二值因素与糖尿病的关联
UCI_BINARY = ["Polyuria", "Polydipsia", "sudden weight loss", "weakness",
              "Polyphagia", "Genital thrush", "visual blurring", "Itching",
              "Irritability", "delayed healing", "partial paresis",
              "muscle stiffness", "Obesity"]
diabetes_assoc = [binary_assoc(uci, c) for c in UCI_BINARY]

# 年龄（连续变量）：Mann-Whitney U 检验
ages_pos = pd.to_numeric(uci.loc[uci["class"] == "Positive", "Age"], errors="coerce").dropna()
ages_neg = pd.to_numeric(uci.loc[uci["class"] == "Negative", "Age"], errors="coerce").dropna()
_, age_p = stats.mannwhitneyu(ages_pos, ages_neg, alternative="two-sided")
diabetes_assoc.append({"factor": "Age",
                       "mean_positive": round(float(ages_pos.mean()), 1),
                       "mean_negative": round(float(ages_neg.mean()), 1),
                       "mannwhitney_p": float(age_p),
                       "significant": bool(age_p < 0.05)})

# 显著且可映射类别的 UCI 因素 → 类别
sig_by_cat = {}
for r in diabetes_assoc:
    cat = UCI_CAT.get(r["factor"])
    if cat and r["significant"]:
        sig_by_cat.setdefault(cat, []).append(r["factor"])

hair_side = set(HAIR_FACTORS)
dia_side = set(sig_by_cat)
jaccard = len(hair_side & dia_side) / len(hair_side | dia_side)
factor_table = [{"factor": f, "hair_sources": h,
                 "diabetes_significant": f in sig_by_cat,
                 "shared": bool(h and f in sig_by_cat)}
                for f, h in HAIR_FACTORS.items()]
shared_significant = [{"factor": f, "uci_columns": sig_by_cat[f],
                       "hair_sources": HAIR_FACTORS[f]}
                      for f in HAIR_FACTORS if f in sig_by_cat]

# ---------- C. 桑基图：风险因素 → 疾病 → 数据源（流量=样本量） ----------
factors = list(HAIR_FACTORS)
sources_h = ["kaggle_hair_health", "luke_hair_loss", "mendeley_survey"]
labels = (factors + ["脱发", "糖尿病"] + sources_h + ["uci_diabetes"])
node_x = [0.01] * len(factors) + [0.5, 0.5] + [0.99] * (len(sources_h) + 1)
node_y = ([0.5 * (i + 0.5) / len(factors) for i in range(len(factors))]
          + [0.3, 0.7] + [0.25, 0.5, 0.75, 0.95])
colors = (["#8ecae6"] * len(factors) + ["#ffb703", "#e76f51"]
          + ["#b8c0c8"] * (len(sources_h) + 1))
fig = go.Figure(go.Sankey(
    arrangement="fixed",
    node=dict(label=labels, x=node_x, y=node_y, color=colors, pad=12, thickness=14),
    link=dict(source=[], target=[], value=[], color="rgba(120,140,160,0.35)"),
))
src_link, tgt_link, val_link = [], [], []
# 因素 → 疾病
for i, f in enumerate(factors):
    h = HAIR_FACTORS[f]
    if h:
        src_link.append(i); tgt_link.append(len(factors)); val_link.append(sum(N[s] for s in h))
    if f in sig_by_cat:  # 糖尿病侧：以实际显著关联为准
        src_link.append(i); tgt_link.append(len(factors) + 1); val_link.append(N["uci_diabetes"])
# 疾病 → 数据源
i_dia_src = len(factors) + 2
for j, s in enumerate(sources_h):
    src_link.append(len(factors)); tgt_link.append(i_dia_src + j); val_link.append(N[s])
src_link.append(len(factors) + 1); tgt_link.append(i_dia_src + 3); val_link.append(N["uci_diabetes"])
fig.update_traces(link=dict(source=src_link, target=tgt_link, value=val_link,
                            color="rgba(120,140,160,0.35)"))
fig.update_layout(title="M6 共同风险因素 → 脱发/糖尿病 → 证据数据源（流量=样本量）",
                  font=dict(size=12), height=560)
fig.write_html(OUT / "m6_sankey.html", include_plotlyjs=True)  # 内嵌 JS，离线可打开

results = {
    "prevalence": {
        "uci_alopecia_rate_positive": round(float(p_pos), 3),
        "uci_alopecia_rate_negative": round(float(p_neg), 3),
        "chi2": round(float(chi2), 2), "p_value": float(pval),
        "odds_ratio": round(float(odds), 2), "odds_ratio_ci95": [round(ci[0], 2), round(ci[1], 2)],
        "background_hair_loss_rate": background,
        "n": {"positive": len(pos), "negative": len(neg), "unique_total": len(uci)},
    },
    "risk_factor_dictionary": factor_table,
    "diabetes_risk_factors": diabetes_assoc,
    "shared_significant_factors": shared_significant,
    "jaccard_overlap": round(jaccard, 3),
    "sankey": {"file": "reports/m6_sankey.html",
               "nodes": len(labels), "links": len(src_link)},
}
(OUT / "m6_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

try:
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
print(f"UCI 真共现(去重后 n={len(uci)}): Alopecia率 Positive={p_pos:.1%} vs Negative={p_neg:.1%}, "
      f"chi2={chi2:.2f}, p={pval:.4f}, OR={odds:.2f} [{ci[0]:.2f},{ci[1]:.2f}]")
print(f"背景脱发率: {background}")
sig_names = [f["factor"] for f in shared_significant]
print(f"共同风险因素: 脱发侧 {len(hair_side)} 项, 糖尿病侧显著 {len(dia_side)} 项, "
      f"共同 {len(shared_significant)} 项 -> {sig_names}, Jaccard={jaccard:.3f}")
print("产物: reports/m6_sankey.html, reports/m6_results.json")
