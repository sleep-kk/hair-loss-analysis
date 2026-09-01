# 脱发影响因素分析与可视化系统

基于 Python 的数据分析项目。围绕「脱发影响因素」这一主题，整合 **4 个公开数据集**，完成 **7 个分析模块**，覆盖从数据画像、单/多因素分析、预测建模、跨数据集验证、跨疾病关联，到交互式风险预测的完整流程。

---

## 一、项目概览

| 模块 | 目标 | 主要数据 | 核心产出 |
|---|---|---|---|
| M1 数据画像与质量评估 | 数据从哪来、质量如何 | 全部 4 数据集 | 数据规模、字段覆盖、缺失值、目标分布 |
| M2 单因素分析 | 谁最容易脱发 | Kaggle + Mendeley + UCI | 年龄/性别/遗传/压力/吸烟等单因素结论 |
| M3 多因素关联 | 因素如何相互作用 | Mendeley | Cramér's V 关联矩阵、交互项检验、高危组合 |
| M4 预测模型 | 输入特征 → 脱发风险 | Kaggle | 5 模型对比、概率校准、阈值调优、SHAP |
| M5 跨数据集验证 | 结论是否可靠 | Kaggle↔Mendeley + Luke | 迁移验证、一致性对比、Luke 时序 LSTM |
| M6 跨疾病关联 | 脱发与糖尿病 | UCI | 患病率对比、共同风险因素、桑基图 |
| M7 交互工具 | 输入特征 → 概率 | M4 模型 | Streamlit 页面 + 离线静态网页 |

---

## 二、数据资产

| 标准文件（`data/raw/`） | 原始来源 | 规模 | 目标变量 | 角色 |
|---|---|---|---|---|
| `kaggle_hair_health.csv` | Predict Hair Fall.csv | 999 × 13 | `hair_loss` (0/1) | 主数据集（字段最全、无缺失） |
| `mendeley_hair_loss_survey.csv` | hairfall_problem3592.csv | 716 × 14 | `do_you_have_hair_fall_problem` (Yes/No) | 补充生活习惯维度 |
| `luke_hair_loss.csv` | Luke_hair_loss_documentation.csv | 400 × 14 | `hair_loss` (Few/Medium/Many/A lot) | 纵向时间序列，用于验证 |
| `uci_diabetes.csv` | diabetes_data_upload.csv | 520 × 17 | `Alopecia`(Yes/No) + `class`(糖尿病) | 跨疾病关联分析 |

---

## 三、目录结构

```
project/
├── data/
│   └── raw/                              # 标准化后的原始数据（4 个数据集）
│       ├── kaggle_hair_health.csv        # 主数据集（999×13）
│       ├── mendeley_hair_loss_survey.csv # 生活习惯调查（716×14）
│       ├── luke_hair_loss.csv            # 纵向时间序列（400×14）
│       └── uci_diabetes.csv              # 糖尿病 + 脱发（520×17）
│
├── models/                               # M4/M5 训练好的模型资产
│   ├── hairloss_xgboost.pkl              # XGBoost（默认主模型）
│   ├── hairloss_random_forest.pkl
│   ├── hairloss_logistic_regression.pkl
│   ├── hairloss_svm.pkl
│   ├── hairloss_mlp.pkl
│   ├── encoders.pkl                      # 特征编码器
│   ├── feature_columns.pkl               # 特征列顺序
│   ├── hairloss_calibrator.pkl           # Platt 概率校准器
│   ├── threshold_config.pkl              # F-beta 判定阈值
│   ├── luke_lstm.pth                     # Luke 时序 LSTM 权重
│   ├── luke_lstm_config.pkl              # LSTM 配置
│   └── model_metrics.csv                 # 5 模型指标汇总
│
├── reports/
│   ├── figures/                          # 27 张分析图表（M1–M5）
│   │   ├── 01_*.png                      # M1 数据画像（4 张）
│   │   ├── 02_*.png                      # M2 单因素（6 张）
│   │   ├── 03_*.png                      # M3 多因素（3 张）
│   │   ├── 04_*.png                      # M4 预测模型（8 张）
│   │   └── 05_*.png                      # M5 跨数据集（6 张）
│   ├── data_quality_report.md            # M1 质量报告
│   ├── 02_univariate_findings.md         # M2 结论
│   ├── 03_multivariate_findings.md       # M3 结论
│   ├── 05_consistency_table.csv          # M5 一致性对比表
│   ├── m6_sankey.html                    # M6 桑基图（内嵌 JS，离线可开）
│   └── m6_results.json                   # M6 统计结果
│
├── 01_data_profile.py                    # M1 数据画像与质量评估
├── 02_univariate_analysis.py             # M2 单因素分析
├── 03_multivariate_analysis.py           # M3 多因素关联分析
├── 04_prediction_model.py                # M4 预测模型
├── 05_cross_validation.py                # M5 跨数据集验证
├── 05_luke_lstm.py                       # M5 Luke 时序 LSTM
├── 06_cross_disease.py                   # M6 跨疾病关联
├── 07_app.py                             # M7 Streamlit 交互页面
├── index.html                            # 离线静态展示页（M1–M6）
└── 项目任务说明.md                        # 任务分工与约定
```



---

## 四、运行方式

**环境依赖**

```bash
pip install pandas numpy matplotlib scipy scikit-learn xgboost shap streamlit plotly torch
```

**各模块（按需运行）**

```bash
python 01_data_profile.py          # M1 数据画像
python 02_univariate_analysis.py   # M2 单因素分析
python 03_multivariate_analysis.py # M3 多因素关联
python 04_prediction_model.py      # M4 预测模型（重新训练 + 保存模型）
python 05_cross_validation.py      # M5 跨数据集验证
python 05_luke_lstm.py             # M5 Luke 时序 LSTM
python 06_cross_disease.py         # M6 跨疾病关联
```

**M7 交互页面（Streamlit）**

```bash
python -m streamlit run 07_app.py
```

浏览器访问 `http://localhost:8501`，输入 9 个特征即可获得脱发概率、风险判定、群体基线对比、高危组合提示与 SHAP 个体解释。

**离线静态展示（无需任何依赖）**

直接双击 `index.html`，即可离线浏览 M1–M6 的全部图表与桑基图（图已内嵌，无需联网）。交付时保持 `index.html` 与 `reports/` 目录的相对结构不变，整体拷贝即可。

---

## 五、核心结论

- **M2 单因素**：压力、水质、家族史、睡眠障碍、贫血与脱发显著相关（关键因素 Yes 组脱发率明显更高）。
- **M3 多因素**：关联最强为压力（V=0.504）与水质（V=0.492）；交互最显著为「睡眠障碍 × 护发产品」（p=0.0036）；高危组合「水质 + 压力 + 家族史 + 睡眠障碍 + 贫血」脱发率高达 **98.97%**。
- **M4 预测模型**：XGBoost 综合最优（CV-AUC 0.878、准确率 0.832），配合 Platt 概率校准与 F-beta 阈值（β=1.5，阈值 0.255）后输出校准概率。
- **M5 跨数据集**：Kaggle↔Mendeley 迁移验证结论方向一致；Luke 纵向验证压力/熬夜上升与脱发加重趋势相符。
- **M6 跨疾病**：UCI 去重后（n=251），糖尿病组脱发率（29.5%）反而低于非糖尿病组（50.0%，OR=0.42，p=0.003）；「肥胖与体重」（体重骤降，OR=7.08）是脱发与糖尿病共享且显著的风险因素。
- **M7 交互**：整合 M1–M5 成果，支持 5 模型切换、校准概率、阈值判定与单样本 SHAP 解释。

---

## 六、离线/非本地交付

本项目提供两种离线交付方式：

1. **静态网页**（`index.html`）：纯前端展示 M1–M6 全部图表，双击即开，零依赖。
2. **交互应用**（`07_app.py`）：本地 Streamlit 服务，支持实时预测（需 Python 环境）。

如需完全脱离 Python 环境运行交互应用，可用 PyInstaller 将 `07_app.py` 与其依赖打包为可执行文件（本项目已提供静态网页作为首选离线方案）。
