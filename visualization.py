# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt
import seaborn as sns # for correlation heatmap
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor


train_df = pd.read_csv("dataset/recipes_train.csv")
test_df = pd.read_csv("dataset/recipes_test.csv")

# print(train_df.shape)
# print(train_df.head(5))
# print(train_df.describe())

# 按照食谱产地进行分组统计
food_count = {}
for _,group in train_df.groupby("cuisine"):
    location = group["cuisine"].head(1).item()
    food_count[location] = {}
    for col in group.columns:
        if col not in ["id", "cuisine"]:
            food_count[location][col] = group[col].sum()
# print(food_count.keys())
# print(food_count)

# 绘制统计结果

import matplotlib.pyplot as plt

# plt.figure()

# subplot_count = len(food_count.keys())

# for i in range(subplot_count):
#     location = list(food_count.keys())[i]
#     plt.subplot(subplot_count, 1, i+1)
#     plt.bar(range(len(food_count[location].keys())), food_count[location].values())
#     plt.title(location)

# # 避免标题和子图重叠
# plt.tight_layout()
# 绘制
# plt.savefig('screenshot/simple_statistics.jpg')

# train_data = train_df.values
# test_data = test_df.values
# print(train_data.shape)

train_df1=train_df.loc[:, (train_df == 0).all(axis=0)]
# print(train_df1.values.shape)


# 拆分训练数据
train_x = train_df.drop(columns=["cuisine"]).values
train_y = train_df["cuisine"].values
le = LabelEncoder()
train_y = le.fit_transform(train_y)


# 分割训练集验证集
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

# print(X_train)
# print(y_train)

xgb = XGBRegressor(n_estimators=100)
xgb.fit(X_train, y_train)
sorted_idx = xgb.feature_importances_.argsort()
# sorted_idx = sorted_idx[-50:]
plt.figure(figsize=(19.2,10.24))
plt.ylabel('Feature', font={'family':'Arial', 'size':10})
plt.xlabel('Xgboost Feature Importance', font={'family':'Arial', 'size':16})
plt.yticks(fontsize = 8)
plt.barh(train_df.columns.values[sorted_idx], xgb.feature_importances_[sorted_idx])

plt.savefig('screenshot/feature_importance_all.jpg')



