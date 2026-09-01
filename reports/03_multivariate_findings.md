# 模块3：多因素关联分析结论文档

- 数据：Mendeley 问卷（清洗后 715 条，总体脱发率 67.4%）
- 目标变量：Hair Loss（0/1）

## 1. 特征关联矩阵
对 9 个特征（年龄分箱）+ 目标计算 Cramér's V。与脱发关联最强的因素：

- Stress: Cramér's V = 0.504
- Water Quality: Cramér's V = 0.492
- Stay Up Late: Cramér's V = 0.440
- Family History: Cramér's V = 0.432
- Hair Products: Cramér's V = 0.397
- Sleep Disturbance: Cramér's V = 0.390
- Age Group: Cramér's V = 0.360
- Anemia: Cramér's V = 0.230
- Gender: Cramér's V = 0.090

## 2. 交互显著性（逻辑回归交互项）
7 个 Yes/No 特征两两建立 Logit 交互模型，交互项 p 值最小的组合如下（p<0.05 视为交互显著）：

| 因素 A | 因素 B | 交互项系数 | p 值 | 是否显著 |
|---|---|---|---|---|
| Sleep Disturbance | Hair Products | -1.120 | 0.0036 | 显著 |
| Stay Up Late | Sleep Disturbance | -1.271 | 0.0047 | 显著 |
| Sleep Disturbance | Stress | -1.229 | 0.0048 | 显著 |
| Sleep Disturbance | Water Quality | -1.074 | 0.0065 | 显著 |
| Water Quality | Anemia | -1.086 | 0.0166 | 显著 |
| Family History | Sleep Disturbance | -0.981 | 0.0180 | 显著 |
| Hair Products | Stress | -1.013 | 0.0205 | 显著 |
| Stay Up Late | Water Quality | -0.923 | 0.0234 | 显著 |
| Hair Products | Anemia | -0.974 | 0.0284 | 显著 |
| Stay Up Late | Stress | -0.919 | 0.0313 | 显著 |

## 3. 高危组合
Top5 特征组合中脱发率最高的组合（仅保留样本量足够的组合）：

| 组合 | 样本量 | 脱发率 |
|---|---|---|
| Water Quality + Stress + Family History + Sleep Disturbance + Anemia | 97 | 99.0% |
| Water Quality + Stress + Family History | 65 | 92.3% |
| Water Quality + Stress + Family History + Sleep Disturbance | 138 | 90.6% |
| Family History + Sleep Disturbance | 13 | 76.9% |
| Stress + Family History + Sleep Disturbance + Anemia | 21 | 76.2% |
| Stress + Sleep Disturbance | 12 | 75.0% |
| Water Quality + Family History | 12 | 75.0% |
| Water Quality + Stress + Sleep Disturbance | 27 | 74.1% |
| Stress + Family History + Anemia | 19 | 73.7% |
| Water Quality + Stress + Family History + Anemia | 30 | 73.3% |

## 4. 结论
（此段在脚本运行后补充关键发现，见控制台输出。）
