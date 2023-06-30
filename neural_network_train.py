import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch import distributions, nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h1,num_h2,num_h3, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h1)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h1, num_h2) 
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h2, num_h3) 
        self.relu3 = torch.nn.ReLU()
        self.linear4 = torch.nn.Linear(num_h3, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)
        return x
    
class Data(Dataset): #构造自己的类，继承自Dataset类
    def __init__(self, data):
        self.data = data
        self.x_data = torch.from_numpy(data[:,:-1])#取每一行，取到最后一列
        self.y_data = torch.from_numpy(data[:,-1])#取每一行，取最后一列
        # self.transform = transform
 
    def __getitem__(self, index: int):
        x = self.x_data[index]
        y = self.y_data[index]
        # if self.transform is not None:
        #     x = self.transform(x)
        #     y = self.transform(y)
        return x, y
 
    def __len__(self):
        return len(self.data)

def load_data():
    train_df = pd.read_csv("dataset/recipes_train.csv")
    test_df = pd.read_csv("dataset/recipes_test.csv")

    train_x = train_df.drop(columns=["cuisine"]).values
    train_y = train_df["cuisine"].values
    le = LabelEncoder()
    train_y = le.fit_transform(train_y)

    X_train, X_valid, y_train, y_valid = train_test_split(train_x, train_y, test_size=0.2, random_state=42)

    data_train = np.hstack((X_train,y_train.reshape(-1,1)))
    data_test = np.hstack((X_valid,y_valid.reshape(-1,1)))

    print('------------data.shape----------')
    print(data_train.shape)


    dataset = Data(data_train)
    dataset_test = Data(data_test)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=32, shuffle=True)
    
    return X_train, X_valid, y_train, y_valid, dataloader, dataloader_test


def train(model,dataloader,train_x):

    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # 设置迭代次数
    epochs = 100
    for epoch in range(epochs):
        sum_loss = 0
        train_correct = 0
        for data in dataloader:
            # 获取数据和标签
            inputs, labels = data  # inputs 维度：[64,1,28,28]
            # 将输入数据展平为一维向量
            inputs = torch.flatten(inputs, start_dim=1)  # 展平数据，转化为[64,784]
            inputs = inputs.to(torch.float32)
            # 计算输出
            outputs = model(inputs)
            # 将梯度清零
            optimizer.zero_grad()
            # 计算损失函数
            loss = cost(outputs, labels)
            # 反向传播计算梯度
            loss.backward()
            # 使用优化器更新模型参数
            optimizer.step()

            # 返回 outputs 张量每行中的最大值和对应的索引，1表示从行维度中找到最大值
            _, id = torch.max(outputs.data, 1)
            # 将每个小批次的损失值 loss 累加，用于最后计算平均损失
            sum_loss += loss.data
            # 计算每个小批次正确分类的图像数量
            train_correct += torch.sum(id == labels.data)
        print('[%d/%d] loss:%.3f, correct:%.3f%%' %
              (epoch + 1, epochs, sum_loss / len(dataloader),
               100 * train_correct / len(train_x),
               ))
    model.eval()

# 测试模型
def test(model, test_loader):
    model.eval()
    test_correct = 0
    for data in test_loader:
        inputs, lables = data
        inputs, lables = Variable(inputs).cpu(), Variable(lables).cpu()
        inputs = torch.flatten(inputs, start_dim=1)  # 展并数据
        inputs = inputs.to(torch.float32)
        outputs = model(inputs)
        _, id = torch.max(outputs.data, 1)
        test_correct += torch.sum(id == lables.data)
    print(f'Accuracy on test set: {100 * test_correct / len(train_y):.3f}%')


num_i = 384  # 输入层节点数
num_h1 = 512  # 隐含层节点数
num_h2 = 256
num_h3 = 128
num_o = 64  # 输出层节点数
batch_size = 64

if __name__ == '__main__':
    X_train, X_valid, y_train, y_valid, dataloader, dataloader_test = load_data()
    model = MLP(num_i, num_h1,num_h2, num_h3,num_o)
    train(model,dataloader,X_train)
    test(model, dataloader_test)