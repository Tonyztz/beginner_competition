# 导入必要的库
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn import manifold, datasets
from sklearn import svm
import joblib

def create_train_test_dataset():
    train_df = pd.read_csv("dataset/recipes_train.csv")
    test_df = pd.read_csv("dataset/recipes_test.csv")
    train_x = train_df.drop(columns=["cuisine"]).values
    train_y = train_df["cuisine"].values
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)

    # 分割训练集验证集
    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid

def sklearn_LogisticRegression(X_train, X_valid, y_train, y_valid):
    # 使用lr模型
    model = LogisticRegression(penalty="l1", C=0.5, solver="liblinear")

    # 训练模型
    model.fit(X_train, y_train)
    print("train accuracy:", accuracy_score(model.predict(X_train), y_train))
    print("valid accuracy:", accuracy_score(model.predict(X_valid), y_valid))
    joblib.dump(model, 'saved_model/LogisticRegression.pkl')
    print('LogisticRegression done')

def sklearn_svm(X_train, X_valid, y_train, y_valid):
    model = svm.SVC(gamma='scale', C=1.0, decision_function_shape='ovr', kernel='rbf')
    # 进行训练
    model.fit(X_train, y_train)
    print("train accuracy:", accuracy_score(model.predict(X_train), y_train))
    print("valid accuracy:", accuracy_score(model.predict(X_valid), y_valid))
    joblib.dump(model, 'saved_model/svm.pkl')
    print('svm done')

def sklearn_decision_tree(X_train, X_valid, y_train, y_valid):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth =3, random_state = 42)  # 初始化模型
    model.fit(X_train, y_train)  # 训练模型
    # 计算每个特征的重要程度
    # print(model.feature_importances_)
    print("train accuracy:", accuracy_score(model.predict(X_train), y_train))
    print("valid accuracy:", accuracy_score(model.predict(X_valid), y_valid))
    joblib.dump(model, 'saved_model/decision_tree.pkl')
    print('decision tree done')



if __name__ == '__main__':
    X_train, X_valid, y_train, y_valid = create_train_test_dataset()
    # sklearn_LogisticRegression(X_train, X_valid, y_train, y_valid)
    # sklearn_svm(X_train, X_valid, y_train, y_valid)
    sklearn_decision_tree(X_train, X_valid, y_train, y_valid)