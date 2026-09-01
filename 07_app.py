# -*- coding: utf-8 -*-
"""
07_app.py —— 模块7：脱发风险预测交互页面（Streamlit）
====================================================
输入 9 个特征 → 输出脱发概率 + 风险判定 + 群体基线对比 + 高危组合提示 + SHAP 个体解释。

整合前几个模块的成果：
    - M1 数据画像：页面顶部展示数据来源、样本量、正负比例
    - M2 单因素：群体基线对比（关键因素 Yes/No 组脱发率）
    - M3 多因素：高危因素组合提示
    - M4 预测模型：5 模型切换 + 指标对比 + Platt 校准 + 阈值判定 + SHAP
    - M5 跨数据集：模型可信度说明

依赖 M4 导出的模型资产（models/）。界面文字中文，代码注释中文，模型/字段名英文。
运行方式：python -m streamlit run 07_app.py
"""
import matplotlib

matplotlib.use("Agg")  # 无界面环境绘图

import re
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# 中文字体配置（SHAP 个体解释图使用中文特征名，需显式指定支持中文的字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans SC"]
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "raw"
MODEL_DIR = BASE_DIR / "models"

# 特征中文名（顺序与 encoders.pkl 的 feature_columns 一致）
FEATURE_CN = {
    "what_is_your_age": "年龄",
    "what_is_your_gender": "性别",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue": "家族史",
    "do_you_stay_up_late_at_night": "熬夜",
    "do_you_have_any_type_of_sleep_disturbance": "睡眠障碍",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems": "水质问题",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair": "化学品/染发",
    "do_you_have_anemia": "贫血",
    "do_you_have_too_much_stress": "压力",
}

# 7 个 Yes/No 特征（除年龄、性别外）
YES_NO_COLS = [
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_stay_up_late_at_night",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair",
    "do_you_have_anemia",
    "do_you_have_too_much_stress",
]

# M4 特征重要性 Top（高危因素，用于提示）
HIGH_RISK_COLS = [
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_have_too_much_stress",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_have_anemia",
]

# 群体基线对比的 6 个关键因素（M2）
KEY_FACTORS = [
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_have_too_much_stress",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_have_anemia",
    "do_you_stay_up_late_at_night",
]

# 5 个模型的保存文件名映射（M4）
MODEL_SLUGS = {
    "XGBoost": "xgboost",
    "Random Forest": "random_forest",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm",
    "MLP (Deep Learning)": "mlp",
}

# M3 结论中的高危组合（因素名 + 该人群脱发率）
RISK_COMBINATIONS = [
    (["水质问题", "压力", "家族史", "睡眠障碍", "贫血"], 98.97),
    (["水质问题", "压力", "家族史"], 92.31),
    (["水质问题", "压力", "家族史", "睡眠障碍"], 90.58),
    (["家族史", "睡眠障碍"], 76.92),
    (["压力", "家族史", "睡眠障碍", "贫血"], 76.19),
]


# ----------------------------------------------------------------------------
# 资产加载（缓存）
# ----------------------------------------------------------------------------
@st.cache_resource
def load_assets():
    """加载 5 个模型 + 编码器 + 校准器 + 阈值配置。"""
    enc = joblib.load(MODEL_DIR / "encoders.pkl")
    cal = joblib.load(MODEL_DIR / "hairloss_calibrator.pkl")
    thr = joblib.load(MODEL_DIR / "threshold_config.pkl")
    models = {name: joblib.load(MODEL_DIR / f"hairloss_{slug}.pkl")
              for name, slug in MODEL_SLUGS.items()}
    return models, enc, cal, thr


def _clean_mendeley(df: pd.DataFrame) -> pd.DataFrame:
    """Mendeley 数据清洗（列名规范化 + 错别字 + 年龄异常值）。"""
    df = df.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
                  for c in df.columns]
    df = df.replace({"Yea": "Yes", "\\No": "No"})
    df = df[df["what_is_your_age"] <= 100]
    return df


@st.cache_data
def load_summary() -> dict:
    """计算数据概览（M1）与群体基线（M2）。"""
    df = _clean_mendeley(pd.read_csv(DATA_DIR / "mendeley_hair_loss_survey.csv"))
    target = df["do_you_have_hair_fall_problem"].map({"No": 0, "Yes": 1})
    n_total = len(df)
    n_pos = int(target.sum())
    baseline = {}
    for col in KEY_FACTORS:
        yes_mask = df[col] == "Yes"
        no_mask = df[col] == "No"
        baseline[col] = {
            "yes_rate": float(target[yes_mask].mean()),
            "no_rate": float(target[no_mask].mean()),
            "yes_n": int(yes_mask.sum()),
            "no_n": int(no_mask.sum()),
        }
    return {
        "n_total": n_total,
        "n_pos": n_pos,
        "n_neg": n_total - n_pos,
        "pos_rate": n_pos / n_total,
        "baseline": baseline,
    }


@st.cache_data
def load_metrics() -> pd.DataFrame:
    """加载 5 模型指标对比表（M4）。"""
    return pd.read_csv(MODEL_DIR / "model_metrics.csv")


# ----------------------------------------------------------------------------
# 预测与解释
# ----------------------------------------------------------------------------
def build_input(age, gender, yes_no, enc) -> pd.DataFrame:
    """按 feature_columns 顺序构造单行特征矩阵（年龄标准化，其余映射为 0/1）。"""
    row = {}
    age_df = pd.DataFrame([[float(age)]], columns=["what_is_your_age"])
    row["what_is_your_age"] = float(enc["age_scaler"].transform(age_df)[0][0])
    row["what_is_your_gender"] = enc["gender_mapping"][gender]
    for col in YES_NO_COLS:
        row[col] = enc["binary_mappings"][col][yes_no[col]]
    return pd.DataFrame([row])[enc["feature_columns"]]


def make_prediction(model, enc, age, gender, yes_no):
    """返回 (特征矩阵, 模型原始概率)。"""
    X = build_input(age, gender, yes_no, enc)
    raw = float(model.predict_proba(X)[0][1])
    return X, raw


def make_shap(model, X, enc, age):
    """对单个样本计算 SHAP 值（仅适用于树模型）。"""
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X)
    if isinstance(sv, list):            # 二分类旧版返回 [neg, pos]
        sv = sv[1]
    expected = explainer.expected_value
    expected = float(expected[1]) if isinstance(expected, (list, np.ndarray)) else float(expected)
    disp = X.values[0].copy()
    disp[0] = float(age)                # 年龄回显原始值
    cn_names = [FEATURE_CN[c] for c in enc["feature_columns"]]
    return sv[0], expected, disp, cn_names


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="脱发风险预测", layout="centered")
    st.title("脱发风险预测系统")
    st.caption("基于 Mendeley 问卷数据训练的多模型对比 + Platt 概率校准 + SHAP 个体解释")

    models, enc, cal, thr = load_assets()
    summary = load_summary()
    metrics = load_metrics()

    # ---- P1 数据概览（M1）----
    st.markdown("### 数据概览")
    st.markdown(
        f"模型基于 **Mendeley 问卷数据** 训练：共 **{summary['n_total']}** 条样本，"
        f"其中脱发 **{summary['n_pos']}** 人、未脱发 **{summary['n_neg']}** 人，"
        f"总体脱发率 **{summary['pos_rate'] * 100:.1f}%**。"
    )
    st.divider()

    # ---- 侧边栏输入 ----
    st.sidebar.header("请输入您的信息")
    age = st.sidebar.slider("年龄", 10, 80, 24)
    gender = st.sidebar.radio("性别", ["Female", "Male"],
                              format_func=lambda x: "女" if x == "Female" else "男")
    yes_no = {}
    for col in YES_NO_COLS:
        yes_no[col] = st.sidebar.radio(
            FEATURE_CN[col], ["No", "Yes"],
            format_func=lambda x: "否" if x == "No" else "是",
            key=col, index=0)

    # ---- P2 多模型切换（M4）----
    model_name = st.sidebar.selectbox("选择模型", list(MODEL_SLUGS.keys()), index=0)
    model = models[model_name]

    if st.button("分析我的脱发风险", type="primary"):
        X, raw = make_prediction(model, enc, age, gender, yes_no)
        result = {
            "X": X, "raw": raw, "model_name": model_name,
            "age": age, "gender": gender, "yes_no": yes_no,
            "threshold": float(thr["threshold"]),
        }
        # 校准与 SHAP 仅对默认 XGBoost 启用（校准器基于 XGBoost OOF 概率拟合）
        if model_name == "XGBoost":
            result["calibrated"] = float(cal.predict_proba([[raw]])[0][1])
            sv, expected, disp, cn_names = make_shap(model, X, enc, age)
            result.update({"sv": sv, "expected": expected,
                           "disp": disp, "cn_names": cn_names})
        st.session_state["result"] = result

    if "result" not in st.session_state:
        st.info("请在左侧填写信息后，点击「分析我的脱发风险」按钮。")
        return

    res = st.session_state["result"]

    # ---- 结果卡片 ----
    col1, col2, col3 = st.columns(3)
    if "calibrated" in res:
        col1.metric("脱发概率（校准）", f"{res['calibrated'] * 100:.1f}%")
    else:
        col1.metric("脱发概率（原始）", f"{res['raw'] * 100:.1f}%")
    col2.metric("模型原始分数", f"{res['raw'] * 100:.1f}%")
    col3.metric("判定阈值", f"{res['threshold'] * 100:.1f}%")

    if "calibrated" in res:
        st.progress(min(res["calibrated"], 1.0))
        if res["calibrated"] >= res["threshold"]:
            st.error(f"结论：有脱发风险（概率 {res['calibrated'] * 100:.1f}% ≥ 阈值 {res['threshold'] * 100:.1f}%）")
        else:
            st.success(f"结论：风险较低（概率 {res['calibrated'] * 100:.1f}% < 阈值 {res['threshold'] * 100:.1f}%）")
    else:
        st.progress(min(res["raw"], 1.0))
        st.info(f"当前模型「{res['model_name']}」未启用概率校准，以上为原始预测分数；"
                f"校准与阈值判定仅适用于默认 XGBoost。")

    # ---- P1 群体基线对比（M2）----
    st.markdown("### 群体基线对比")
    rows = []
    for col in KEY_FACTORS:
        b = summary["baseline"][col]
        rows.append({
            "因素": FEATURE_CN[col],
            "你的选择": "是" if res["yes_no"][col] == "Yes" else "否",
            "Yes 组脱发率": f"{b['yes_rate'] * 100:.1f}%",
            "No 组脱发率": f"{b['no_rate'] * 100:.1f}%",
        })
    st.dataframe(pd.DataFrame(rows))

    # ---- P2 高危组合提示（M3）----
    st.markdown("### 高危组合提示")
    yes_cn = {FEATURE_CN[c] for c in YES_NO_COLS if res["yes_no"][c] == "Yes"}
    matched = [(factors, rate) for factors, rate in RISK_COMBINATIONS
               if set(factors).issubset(yes_cn)]
    if matched:
        for factors, rate in matched:
            st.warning(f"命中高危组合「{' + '.join(factors)}」：该人群脱发率约 {rate:.0f}%")
    else:
        st.info("未命中已知的高危因素组合。")

    # ---- SHAP 个体解释（仅 XGBoost）----
    if "sv" in res:
        st.markdown("### 为什么是这个结果？（SHAP 个体解释）")
        try:
            exp = shap.Explanation(values=res["sv"], base_values=res["expected"],
                                   data=res["disp"], feature_names=res["cn_names"])
            shap.plots.waterfall(exp, show=False, max_display=9)
            st.pyplot(plt.gcf(), clear_figure=True)
        except Exception as e:  # SHAP 失败不阻断主流程
            st.info(f"SHAP 解释暂不可用：{e}")

    # ---- P2 多模型指标对比（M4）----
    st.markdown("### 模型指标对比")
    fmt = metrics.copy()
    for c in ["cv_auc", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        fmt[c] = fmt[c].round(4)
    st.dataframe(fmt)

    # ---- P1 可信度说明（M5）----
    st.markdown("### 模型可信度说明")
    xgb_auc = metrics.loc[metrics["model"] == "XGBoost", "roc_auc"].values[0]
    st.markdown(
        f"- 最优模型 XGBoost，测试集 ROC-AUC = **{xgb_auc:.3f}**\n"
        f"- 概率经 **Platt Scaling** 校准，决策阈值按 F-1.5 调优（约 {res['threshold']:.2f}）\n"
        f"- 跨数据集迁移验证：模型迁移到合成数据（Kaggle）上失效（AUC≈0.46），"
        f"说明结论仅适用于 **Mendeley 类真实问卷人群**"
    )


if __name__ == "__main__":
    main()
