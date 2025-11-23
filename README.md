# My-Kaggle_Competitions

> 记录我在 Kaggle 的练习与参赛过程：从 EDA、特征工程、建模到提交与复盘。  
> This repo is a living logbook of my Kaggle journey: datasets, notebooks, models, and lessons learned.

## 🎯 仓库目标 | Purpose

- **记录过程**：从问题理解 → 数据探索（EDA）→ 特征工程 → 训练与验证 → 提交与复盘。  
- **可复现**：统一的目录结构、环境与脚本，保证“换台机器也能跑”。  
- **知识沉淀**：把踩坑与经验写清楚，复用在下一个比赛。  
- **快速起步**：提供「一键模板」添加新比赛。(后续熟练后想做，帮助大家快速入门)

---

## 🗂️ 仓库结构 | Repository Layout

建议按比赛创建子目录，每个子目录自带自己的 `README` 与脚本：

```
My-Kaggle_Competitions/
├─ competitions/
│  ├─ house-prices/                # 示例：每个比赛一个文件夹
│  │  ├─ notebooks/                # EDA 与建模 Notebook
│  │  ├─ src/                      # 可复用的训练/推理脚本
│  │  ├─ models/                   # 训练好的模型(必要时用 Git LFS)
│  │  ├─ submissions/              # 提交文件与分数
│  │  ├─ configs/                  # 超参/路径等 YAML
│  │  └─ README.md                 # 本比赛说明、结果、复盘
├─ templates/
│  └─ competition-template/        # 新比赛的骨架模板（拷贝即用）
├─ tools/                          # 小工具：数据下载、特征检查、CV 可视化等
├─ .gitignore
└─ README.md                       # 本文件（总览）
```
现阶段暂时无法做到如此详细，算是一个小的憧憬，干净利落的仓库结构，后续会持续改进更新。

---

## 🧱 比赛工作流 | Standard Workflow

1. **理解任务**：目标/指标/限制（时间、内存、外部数据是否允许）。  
2. **EDA**：  
   - 缺失值分布、异常值、数值分布（偏态/峰度）、类别基数；  
   - 目标变量特性（是否对数化）；  
   - 数据泄漏排查（ID、时间、后验特征）。  
3. **特征工程**：  
   - 规则修复（例如 “缺失=没有该设施”）；  
   - 数值转换（Yeo-Johnson/对数、标准化）、One-Hot/目标编码；  
   - 交互/比率/统计特征；保留特征字典与生成脚本。  
4. **验证策略（CV）**：  
   - KFold/StratifiedKFold/GroupKFold/时间序列 Split；  
   - **先确认 CV 和 Public LB 一致性**，再调参。  
5. **模型与融合**：  
   - 线性（Ridge/Lasso/ENet）、树模型（XGBoost/LightGBM/CatBoost）、简单 NN；  
   - Stacking/加权平均/Rank Averaging。  
6. **提交与复盘**：  
   - `submissions/` 保存文件名+分数+备注；  


## 🙌 致谢 | Acknowledgements

Kaggle 社区 & 开源工具（pandas / numpy / matplotlib / seaborn / scikit-learn / XGBoost / LightGBM / CatBoost…）。

---

> 如果你喜欢这个结构，也欢迎 Star / Fork；如有问题或建议，提问题我会跟进。
