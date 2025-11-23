# House Prices – Advanced Regression（我的第一个 Kaggle 解答）

只有一个 Notebook：`getting-started-with-house-prices-advanced-regre.ipynb`  
目标：最小可行解，先跑通提交流程，把思路讲清楚。

## 一些自己学到的处理思路，以及一个很简单的实现

- 读取数据：`pandas.read_csv()`  
  用最直接的方式拿到 `train.csv / test.csv`，开始清洗与建模。

- 目标变换：`np.log1p(SalePrice)`  
  对目标做对数转换，缓解房价分布的右偏，让误差更接近正态，也贴合竞赛的 log-RMSE 评测。  
  （预测后用 `np.expm1()` 还原回真实价格）

- 缺失值处理：`fillna()` 或按列用中位数/众数/“None”  
  简单可靠，避免丢样本或让模型报错；先求稳，再考虑精细规则。

- 类别编码：`pd.get_dummies()`  
  把离散特征变成 0/1 列，便于通用回归器使用，不强加“大小关系”的假设。  
  注意对齐 train/test 的列，避免维度不一致。

- 拆分特征与标签：`X, y = features, log1p(SalePrice)`  
  明确训练输入与监督目标，后面直接喂给模型。

- 交叉验证（简易）：`KFold(n_splits=5, shuffle=True, random_state=42)`  
  粗看稳定性与方差，避免偶然好的/坏的单次划分误导判断。

- 基线建模：先上一个简单模型（例如 `Ridge` 或 `RandomForestRegressor`）  
  先拿到一个靠谱起点，验证清洗与编码是否合理，再考虑换/叠更强的模型。

- 预测与还原：`y_pred = np.expm1(y_pred_log)`  
  从对数空间回到真实货币单位，和提交格式一致。

- 生成提交：`DataFrame[['Id','SalePrice']].to_csv('submission.csv', index=False)`  
  两列即可，在 Kaggle 页面上传验证结果。

- 防止数据泄漏（贯穿全程）  
  所有“先拟合再应用”的步骤（均值/中位数、编码器等）只在 **train** 上拟合，再用到 **test**；  
  这样本地验证和公开榜分数才更可信。