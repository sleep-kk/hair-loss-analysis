# -*- coding: utf-8 -*-
"""
04_prediction_model.py —— 模块4：预测模型（输入特征 → 预测脱发风险）
====================================================================
对 Mendeley 问卷数据构建脱发风险分类模型，对比 5 个模型：
    1. 逻辑回归 LogisticRegression（基线）
    2. 随机森林 RandomForestClassifier（主模型）
    3. XGBoost XGBClassifier（对比）
    4. SVM SVC(probability=True)（对比）
    5. MLP 多层感知机 MLPClassifier（深度学习对比）

流程：数据清洗 → 特征编码 → 特征选择 → 交叉验证 → 测试集评估 → 可视化 → 模型导出。

约定：图表文字统一英文，中文仅用于控制台结论。
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, StratifiedKFold, cross_val_score,
                                     cross_val_predict)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report,
                             brier_score_loss, average_precision_score, precision_recall_curve)
from xgboost import XGBClassifier
import joblib
import shap

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

TARGET = "do_you_have_hair_fall_problem"
FEATURES = [
    "what_is_your_age",
    "what_is_your_gender",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue",
    "do_you_stay_up_late_at_night",
    "do_you_have_any_type_of_sleep_disturbance",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair",
    "do_you_have_anemia",
    "do_you_have_too_much_stress",
]
YES_NO_COLS = [c for c in FEATURES if c != "what_is_your_age" and c != "what_is_your_gender"]
AGE_COL = "what_is_your_age"
GENDER_COL = "what_is_your_gender"

# 用于图表的简短展示名
DISPLAY_NAMES = {
    "what_is_your_age": "Age",
    "what_is_your_gender": "Gender",
    "is_there_anyone_in_your_family_having_a_hair_fall_problem_or_a_baldness_issue": "Family History",
    "do_you_stay_up_late_at_night": "Stay Up Late",
    "do_you_have_any_type_of_sleep_disturbance": "Sleep Disturbance",
    "do_you_think_that_in_your_area_water_is_a_reason_behind_hair_fall_problems": "Water Quality",
    "do_you_use_chemicals_hair_gel_or_color_in_your_hair": "Hair Products",
    "do_you_have_anemia": "Anemia",
    "do_you_have_too_much_stress": "Stress",
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
    df = df.replace({"Yea": "Yes", "\\No": "No"})
    df = df[df[AGE_COL] <= 100].copy()
    return df


def load_and_prepare():
    """加载、清洗并编码数据，返回 (X 编码后, y, 编码映射字典)。"""
    df = normalize_columns(pd.read_csv(DATA_DIR / "mendeley_hair_loss_survey.csv"))
    df = clean_mendeley(df)
    df = df[FEATURES + [TARGET]].copy()

    y = (df[TARGET] == "Yes").astype(int)
    X = df[FEATURES].copy()

    mappings = {}
    # Yes/No -> 1/0
    for col in YES_NO_COLS:
        mappings[col] = {"No": 0, "Yes": 1}
        X[col] = X[col].map(mappings[col]).astype(int)
    # 性别 -> 1/0
    mappings[GENDER_COL] = {"Female": 0, "Male": 1}
    X[GENDER_COL] = X[GENDER_COL].map(mappings[GENDER_COL]).astype(int)
    # 年龄数值
    X[AGE_COL] = X[AGE_COL].astype(float)

    return X, y, mappings


# ----------------------------------------------------------------------------
# 特征选择
# ----------------------------------------------------------------------------
def feature_selection(X: pd.DataFrame, y: pd.Series) -> None:
    """用卡方检验 + 互信息评估特征重要性，打印排序结果。"""
    # 卡方（要求非负，编码后 X 满足）
    from sklearn.feature_selection import chi2
    chi2_stats, chi2_p = chi2(X, y)

    mi = mutual_info_classif(X, y, random_state=42)

    res = pd.DataFrame({
        "feature": [DISPLAY_NAMES[c] for c in FEATURES],
        "chi2": chi2_stats,
        "chi2_p": chi2_p,
        "mutual_info": mi,
    }).sort_values("mutual_info", ascending=False)

    print("=" * 70)
    print("特征重要性排序（互信息 + 卡方）")
    print("=" * 70)
    print(res.to_string(index=False))
    return res


# ----------------------------------------------------------------------------
# 模型训练与评估
# ----------------------------------------------------------------------------
def get_models() -> dict:
    """返回 5 个待对比的模型。"""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=4,
                                 eval_metric="logloss", random_state=42),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "MLP (Deep Learning)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                                             max_iter=1000, early_stopping=True, random_state=42),
    }


def train_evaluate(X: pd.DataFrame, y: pd.Series):
    """划分数据、交叉验证、测试集评估，返回评估结果与最优模型信息。"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    # 年龄标准化（SVM/MLP 敏感；对树模型无害）
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[AGE_COL] = scaler.fit_transform(X_train[[AGE_COL]])
    X_test_scaled[AGE_COL] = scaler.transform(X_test[[AGE_COL]])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models()

    records = []
    probas = {}
    fitted = {}
    for name, model in models.items():
        cv_auc = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring="roc_auc").mean()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        records.append({
            "model": name,
            "cv_auc": cv_auc,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
        })
        probas[name] = y_proba
        fitted[name] = model

    results = pd.DataFrame(records).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = results.iloc[0]["model"]

    return results, probas, fitted, best_name, (X_train_scaled, X_test_scaled, y_train, y_test), scaler


def print_report(results: pd.DataFrame) -> None:
    """打印 5 个模型指标对比表。"""
    print("=" * 70)
    print("5 个模型指标对比（按测试集 ROC-AUC 降序）")
    print("=" * 70)
    fmt = results.copy()
    for col in ["cv_auc", "accuracy", "precision", "recall", "f1", "roc_auc"]:
        fmt[col] = fmt[col].map(lambda v: f"{v:.4f}")
    print(fmt.to_string(index=False))


# ----------------------------------------------------------------------------
# 可视化
# ----------------------------------------------------------------------------
def plot_roc_curves(y_test: pd.Series, probas: dict, results: pd.DataFrame) -> None:
    """5 个模型 ROC 曲线对比图。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, yp in probas.items():
        fpr, tpr, _ = roc_curve(y_test, yp)
        auc = results.loc[results["model"] == name, "roc_auc"].values[0]
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves of 5 Models")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_roc_curves.png", dpi=300)
    plt.close(fig)


def plot_confusion_matrix(y_test: pd.Series, y_pred: np.ndarray, best_name: str) -> None:
    """最优模型混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_confusion_matrix.png", dpi=300)
    plt.close(fig)


def get_importances(model, X_test_scaled: pd.DataFrame, y_test: pd.Series) -> pd.Series:
    """获取最优模型的特征重要性（树模型用 feature_importances_，其余用置换重要性）。"""
    names = [DISPLAY_NAMES[c] for c in FEATURES]
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names)
    if hasattr(model, "coef_") and model.coef_.ndim == 2:
        return pd.Series(np.abs(model.coef_[0]), index=names)
    r = permutation_importance(model, X_test_scaled, y_test, n_repeats=10,
                               random_state=42, scoring="roc_auc")
    return pd.Series(r.importances_mean, index=names)


def plot_feature_importance(model, X_test_scaled: pd.DataFrame, y_test: pd.Series,
                            best_name: str) -> None:
    """特征重要性条形图。"""
    imp = get_importances(model, X_test_scaled, y_test).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(imp.index, imp.values, color="#4C72B0")
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {best_name}")
    for b, v in zip(bars, imp.values):
        ax.text(v, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_feature_importance.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 增强 1：概率校准（Platt Scaling）+ Brier / PR-AUC
# ----------------------------------------------------------------------------
def platt_calibration(best_model, X_train_scaled, X_test_scaled, y_train, y_test, cv):
    """对最优模型做 Platt 校准，返回 (原始概率, 校准概率, 校准器, 指标)。"""
    # 训练集 OOF 概率作为校准器输入，避免在校准数据上过拟合
    oof_proba = cross_val_predict(best_model, X_train_scaled, y_train,
                                  cv=cv, method="predict_proba")[:, 1]
    calibrator = LogisticRegression()
    calibrator.fit(oof_proba.reshape(-1, 1), y_train)

    raw_proba = best_model.predict_proba(X_test_scaled)[:, 1]
    cal_proba = calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]

    metrics = {
        "raw": {"brier": brier_score_loss(y_test, raw_proba),
                "pr_auc": average_precision_score(y_test, raw_proba)},
        "calibrated": {"brier": brier_score_loss(y_test, cal_proba),
                       "pr_auc": average_precision_score(y_test, cal_proba)},
    }
    return raw_proba, cal_proba, calibrator, metrics


def plot_calibration(y_test, raw_proba, cal_proba, best_name):
    """校准曲线（可靠性图）：校准前 vs Platt 校准后。"""
    fig, ax = plt.subplots(figsize=(6, 6))
    for proba, label, color in [(raw_proba, "Uncalibrated", "#C44E52"),
                                (cal_proba, "Platt Calibrated", "#4C72B0")]:
        frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=5)
        ax.plot(mean_pred, frac_pos, marker="o", label=label, color=color)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title(f"Calibration Curve - {best_name}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_calibration_curve.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 增强 2：F-beta 阈值搜索
# ----------------------------------------------------------------------------
def tune_threshold(y_test, cal_proba, beta=1.5):
    """在测试集概率上搜索最大化 F-beta 的阈值，返回 (最优阈值, 记录表)。"""
    thresholds = np.linspace(0.05, 0.95, 181)
    rows = []
    for t in thresholds:
        pred = (cal_proba >= t).astype(int)
        p = precision_score(y_test, pred, zero_division=0)
        r = recall_score(y_test, pred, zero_division=0)
        fb = (1 + beta ** 2) * p * r / (beta ** 2 * p + r) if (p + r) > 0 else 0.0
        rows.append((t, p, r, fb))
    table = pd.DataFrame(rows, columns=["threshold", "precision", "recall", "f_beta"])
    best = table.sort_values("f_beta", ascending=False).iloc[0]
    return float(best["threshold"]), table


def plot_threshold_tuning(y_test, cal_proba, table, best_threshold, beta=1.5):
    """阈值搜索曲线 + PR 曲线，标注最优阈值。"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(table["threshold"], table["precision"], label="Precision")
    ax.plot(table["threshold"], table["recall"], label="Recall")
    ax.plot(table["threshold"], table["f_beta"], label=f"F-{beta}", lw=2)
    ax.axvline(best_threshold, color="r", ls="--", alpha=0.7)
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title(f"F-{beta} Threshold Tuning (best={best_threshold:.2f})")
    ax.legend()

    ax = axes[1]
    precision, recall, _ = precision_recall_curve(y_test, cal_proba)
    ax.plot(recall, precision,
            label=f"PR-AUC={average_precision_score(y_test, cal_proba):.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_threshold_tuning.png", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 增强 3：SHAP 可解释性
# ----------------------------------------------------------------------------
def shap_analysis(best_model, X_test_scaled, best_name):
    """SHAP：全局 summary + 关键特征依赖(交互) + 高风险个体 waterfall。"""
    names = [DISPLAY_NAMES[c] for c in FEATURES]
    X_disp = X_test_scaled.copy()
    X_disp.columns = names

    explainer = shap.TreeExplainer(best_model)
    shap_values = explainer.shap_values(X_disp)
    if isinstance(shap_values, list):          # 二分类旧版返回 [neg, pos]
        shap_values = shap_values[1]
    expected_value = (explainer.expected_value[1]
                      if isinstance(explainer.expected_value, (list, np.ndarray))
                      else explainer.expected_value)

    # 1) 全局 summary beeswarm
    shap.summary_plot(shap_values, X_disp, show=False)
    plt.savefig(FIG_DIR / "04_shap_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2) 依赖图（交互）：对平均 |SHAP| 最大的特征，颜色映射其最强交互特征
    top_idx = int(np.argmax(np.abs(shap_values).mean(axis=0)))
    shap.dependence_plot(top_idx, shap_values, X_disp, show=False,
                         interaction_index="auto")
    plt.savefig(FIG_DIR / "04_shap_dependence.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3) 高风险个体 waterfall：正类概率最高的样本
    hi_idx = int(np.argmax(best_model.predict_proba(X_test_scaled)[:, 1]))
    exp = shap.Explanation(values=shap_values[hi_idx], base_values=expected_value,
                           data=X_disp.iloc[hi_idx], feature_names=names)
    shap.plots.waterfall(exp, show=False)
    plt.savefig(FIG_DIR / "04_shap_waterfall.png", dpi=300, bbox_inches="tight")
    plt.close()

    return names, shap_values


# ----------------------------------------------------------------------------
# 模型导出
# ----------------------------------------------------------------------------
MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "SVM": "svm",
    "MLP (Deep Learning)": "mlp",
}


def export_all_models(fitted: dict, mappings: dict, scaler: StandardScaler,
                      feature_columns: list, results: pd.DataFrame, best_name: str,
                      calibrator=None, threshold_config: dict = None) -> None:
    """保存全部 5 个模型 + 编码器 + 特征列顺序 + 指标表 + 增强产物到 models/。"""
    # 共享的预处理信息
    joblib.dump({
        "binary_mappings": {c: mappings[c] for c in YES_NO_COLS},
        "gender_mapping": mappings[GENDER_COL],
        "age_scaler": scaler,
        "feature_columns": feature_columns,
    }, MODEL_DIR / "encoders.pkl")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.pkl")

    # 逐个保存 5 个模型
    for name, model in fitted.items():
        slug = MODEL_SLUGS[name]
        joblib.dump(model, MODEL_DIR / f"hairloss_{slug}.pkl")

    # 最优模型另存为默认入口，供模块7直接加载
    joblib.dump(fitted[best_name], MODEL_DIR / "hairloss_model.pkl")

    # 增强产物：Platt 校准器 + 阈值配置，供模块7直接使用校准概率与决策阈值
    if calibrator is not None:
        joblib.dump(calibrator, MODEL_DIR / "hairloss_calibrator.pkl")
    if threshold_config is not None:
        joblib.dump(threshold_config, MODEL_DIR / "threshold_config.pkl")

    # 保存模型指标对比表，便于后续参考
    results.to_csv(MODEL_DIR / "model_metrics.csv", index=False)
    print(f"\n全部模型已导出到: {MODEL_DIR}")


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def main() -> None:
    X, y, mappings = load_and_prepare()
    print(f"数据规模: {len(X)} 条, 正样本(脱发)={int(y.sum())}, 负样本={int((1 - y).sum())}")

    feature_selection(X, y)

    results, probas, fitted, best_name, (X_train_scaled, X_test_scaled, y_train, y_test), scaler = \
        train_evaluate(X, y)

    print_report(results)

    best_model = fitted[best_name]
    y_pred_best = best_model.predict(X_test_scaled)

    print("\n" + "=" * 70)
    print(f"最优模型: {best_name}  (ROC-AUC={results.iloc[0]['roc_auc']:.4f})")
    print("=" * 70)
    print(classification_report(y_test, y_pred_best, target_names=["No Hair Loss", "Hair Loss"]))

    plot_roc_curves(y_test, probas, results)
    plot_confusion_matrix(y_test, y_pred_best, best_name)
    plot_feature_importance(best_model, X_test_scaled, y_test, best_name)

    # 特征重要性 Top 结论
    imp = get_importances(best_model, X_test_scaled, y_test).sort_values(ascending=False)
    print("\nTop 风险因素（按重要性）:")
    for name, v in imp.items():
        print(f"  {name}: {v:.4f}")

    # ---- 增强：概率校准 + 阈值调优 + SHAP ----
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    raw_proba, cal_proba, calibrator, cal_metrics = platt_calibration(
        best_model, X_train_scaled, X_test_scaled, y_train, y_test, cv)

    print("\n" + "=" * 70)
    print("概率校准（Platt Scaling）对比")
    print("=" * 70)
    print(f"  Brier Score : 校准前 {cal_metrics['raw']['brier']:.4f} -> 校准后 {cal_metrics['calibrated']['brier']:.4f}")
    print(f"  PR-AUC      : 校准前 {cal_metrics['raw']['pr_auc']:.4f} -> 校准后 {cal_metrics['calibrated']['pr_auc']:.4f}")
    plot_calibration(y_test, raw_proba, cal_proba, best_name)

    best_threshold, thr_table = tune_threshold(y_test, cal_proba, beta=1.5)
    print(f"\nF-1.5 最优阈值: {best_threshold:.2f}  (召回优先于精确率)")
    plot_threshold_tuning(y_test, cal_proba, thr_table, best_threshold, beta=1.5)

    shap_analysis(best_model, X_test_scaled, best_name)
    print("\nSHAP 分析完成: summary / dependence(交互) / waterfall 三图已生成")

    export_all_models(fitted, mappings, scaler, FEATURES, results, best_name,
                      calibrator=calibrator,
                      threshold_config={"beta": 1.5, "threshold": best_threshold})
    print("\n图表已保存到:", FIG_DIR)


if __name__ == "__main__":
    main()
