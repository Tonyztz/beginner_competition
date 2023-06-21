import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import random

train_df = pd.read_csv("dataset/recipes_train.csv")
test_df = pd.read_csv("dataset/recipes_test.csv")
train_x = train_df.drop(columns=["cuisine"]).values
train_y = train_df["cuisine"].values
le = LabelEncoder()
train_y = le.fit_transform(train_y)

def subsample(dataset, ratio=0.6):
	sample = list()
	n_sample = round(len(dataset) * ratio)
	while len(sample) < n_sample:
		index = random.randrange(len(dataset))
		sample.append(dataset[index])
	return np.array(sample)


# 分割训练集验证集
X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

model_num = 10
bootstrap_sampling_rate = 0.6
predictions = []
predictions_train = []
predictions_valid = []

for i in range(model_num):
    bag = subsample(np.hstack((X_train, y_train.reshape(-1,1))))
    model =  LogisticRegression(penalty="l1", C=0.5, solver="liblinear")
    model.fit(bag[:,:-1],bag[:,-1])

    predictions.append(model.predict(bag[:,:-1]))
    predictions_train.append(model.predict(X_train))
    predictions_valid.append(model.predict(X_valid))
train_predict = np.zeros((len(X_train)))
valid_predict = np.zeros((len(X_valid)))
print(np.array(predictions_train).shape)
print(np.array(predictions_train)[:,0])
for i in range(X_train.shape[0]):
	train_predict[i] = np.argmax(np.bincount(np.array(predictions_train)[:,i]))

for i in range(X_valid.shape[0]):
	valid_predict[i] = np.argmax(np.bincount(np.array(predictions_valid)[:,i]))


print("train accuracy:", accuracy_score(train_predict, y_train))
print("valid accuracy:", accuracy_score(valid_predict, y_valid))

