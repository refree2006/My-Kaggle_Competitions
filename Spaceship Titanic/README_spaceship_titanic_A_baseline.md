# Spaceship Titanic – Baseline A (EDA + Logistic Regression)

> Kaggle Competition: **Spaceship Titanic**  
> Kaggle Notebook: *(to be filled with your public notebook link)*

本仓库目前实现了 **方案 A：极简基线**，目标是跑通从原始数据到 Kaggle 提交的完整流程，建立一个干净、可靠的起点，便于后续在方案上继续迭代。

---

## 1. 项目简介 / Project Overview

- **任务 / Task**：根据乘客信息预测其是否被 `Transported`（传送到另一维度），本质是一个二分类问题。
- **数据 / Data**：
  - `train.csv`：包含特征和标签 `Transported`
  - `test.csv`：只包含特征，需要我们预测 `Transported`
  - `sample_submission.csv`：官方提交格式示例
- **评估指标 / Metric**：Accuracy（分类准确率）

本基线只使用了含义清晰、预处理简单的一小部分特征，通过 **探索性数据分析（EDA） + 逻辑回归模型** 搭建了首个提交方案。

---

## 2. 方案 A 设计思想 / Design Philosophy

### 2.1 原则

1. **少而精的特征**：先用最干净、直观的一批列，暂时不处理复杂字段（如 `Cabin`, `Name`）。
2. **简单、稳定的预处理**：缺失值采用中位数/众数填充，不引入复杂插补方法。
3. **易解释的模型**：选用 Logistic Regression 和简单 Random Forest 作为起步模型。
4. **完整闭环**：从读取数据 → EDA → 预处理 → 建模 → 验证 → 生成 `submission.csv`，保证无 bug。

### 2.2 使用的特征 / Features Used

**数值特征（Numeric features）**

- `Age`
- `RoomService`
- `FoodCourt`
- `ShoppingMall`
- `Spa`
- `VRDeck`

这些字段表示乘客年龄与在飞船上不同消费项目的消费金额。

**类别/布尔特征（Categorical / Boolean features）**

- `HomePlanet`
- `CryoSleep`
- `Destination`
- `VIP`

这些字段反映乘客的来源星球、是否处于冷冻睡眠、目的地和是否 VIP 身份。

**暂未使用的特征（Later work）**

- `PassengerId`（作为提交用 ID 保留）
- `Cabin`（后续在方案 B 中拆分为 Deck/Side）
- `Name`（暂不使用）

---

## 3. 数据预处理 / Data Preprocessing

预处理通过 `sklearn` 的 `Pipeline` 与 `ColumnTransformer` 实现，确保在训练和预测阶段使用完全一致的流程。

### 3.1 缺失值填充 / Missing Value Imputation

- 数值列（`Age` + 5 个消费列）  
  - 使用 `SimpleImputer(strategy="median")`  
  - 原因：中位数对极端值不敏感，适合长尾分布的消费金额。
- 类别/布尔列（`HomePlanet`, `CryoSleep`, `Destination`, `VIP`）  
  - 使用 `SimpleImputer(strategy="most_frequent")`  
  - 将缺失值填为最常见类别，作为“默认值”。

### 3.2 类别编码 / Categorical Encoding

- 使用 `OneHotEncoder(handle_unknown="ignore")`
- 对上述 4 个类别/布尔特征进行 One-Hot 编码：
  - 例如 `HomePlanet` → `HomePlanet_Earth`, `HomePlanet_Mars`, `HomePlanet_Europa`, ...
  - `handle_unknown="ignore"` 确保测试集中出现新类别时不会报错。

### 3.3 预处理管线 / Preprocessing Pipeline

简化伪代码结构：

```python
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)
```

---

## 4. 模型与训练 / Model & Training

### 4.1 标签处理 / Target

```python
y = train["Transported"].astype(int)  # True/False → 1/0
X = train[feature_cols]
X_test = test[feature_cols]
```

### 4.2 训练/验证划分 / Train–Validation Split

- 使用 `train_test_split` 做 8:2 划分验证集，分层抽样保证标签比例一致。

```python
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)
```

### 4.3 Logistic Regression Baseline

完整模型通过 Pipeline 串联预处理与分类器：

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

log_reg_clf = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression(
        max_iter=1000,
        n_jobs=-1
    )),
])

log_reg_clf.fit(X_train, y_train)
y_valid_pred = log_reg_clf.predict(X_valid)
```

**评估 / Evaluation**

```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

val_acc = accuracy_score(y_valid, y_valid_pred)
cm = confusion_matrix(y_valid, y_valid_pred)
report = classification_report(y_valid, y_valid_pred)
```

- 使用 Accuracy 作为主指标；
- 同时通过混淆矩阵和分类报告观察：
  - 哪一类更容易被误判；
  - Precision / Recall / F1 是否平衡。

> TODO: 在实际仓库中记录一次验证集准确率和首个 Kaggle Public LB 分数。

---

## 5. 后续工作计划 / Future Work

方案 A 是整个项目的起点，后续计划包括：

1. **方案 B：特征工程 + 树模型（主力方案）**
   - Cabin 拆分 Deck / Side / CabinNum
   - 构造 `TotalSpent`, `NoSpend`, 消费占比等特征
   - 加入缺失标记特征（`xxx_is_missing`）
   - 采用 LightGBM / CatBoost 等模型

2. **方案 C：高级编码 + 集成学习**
   - Target Encoding / Frequency Encoding 等高级类别编码
   - 模型融合：LightGBM + XGBoost + CatBoost 加权平均或 Stacking
   - 尝试简单的 Tabular NN 作为对比

3. **实验记录 & 文档化**
   - 在 README 或 `experiments.md` 中记录不同版本（v1/v2/v3）的设置与得分
   - 持续更新：新增特征、模型与 Kaggle 成绩

---

如果你正在查看这个仓库，欢迎提出建议或 issue，一起优化 Spaceship Titanic 的解决方案 🚀
