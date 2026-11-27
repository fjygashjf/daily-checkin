#!/usr/bin/env python
# coding: utf-8

# In[3]:


# ================= 第 13 天 · 单 Cell 完整训练 =================
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

# ---- 1. 数据 ----
trans = transforms.ToTensor()
mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
train_iter = data.DataLoader(mnist_train, batch_size=256, shuffle=True, num_workers=0)
test_iter  = data.DataLoader(mnist_test,  batch_size=256, shuffle=False, num_workers=0)

# ---- 2. 参数（本 Cell 内唯一一次定义）----
W = torch.normal(0, 0.01, size=(784, 10), requires_grad=True)
b = torch.zeros(10, requires_grad=True)

# ---- 3. softmax / 模型 / 损失 ----
def softmax(X):
    X_exp = torch.exp(X)
    part = X_exp.sum(1, keepdim=True)
    return X_exp / part

def net(X):
    return softmax(torch.matmul(X.reshape((-1, 784)), W) + b)

def cross_entropy(y_hat, y):
    return - torch.log(y_hat[range(len(y_hat)), y])

# ---- 4. 准确率 ----
def accuracy(y_hat, y):
    pred = y_hat.argmax(axis=1)
    return float((pred.type(y.dtype) == y).sum())

def evaluate_accuracy(net, data_iter):
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in data_iter:
            correct += accuracy(net(X), y)
            total += y.numel()
    return correct / total

# ---- 5. 训练前自诊断（关键！）----
X0, y0 = next(iter(train_iter))
l0 = cross_entropy(net(X0), y0).sum()
l0.backward()
print("【诊断1】首个batch损失:", l0.item())          # 应约 2.3（= -log0.1）
print("【诊断2】W.grad 非零?", W.grad.abs().sum().item() > 0)   # 应为 True
W.grad.zero_(); b.grad.zero_()

# ---- 6. 正式训练 ----
trainer = torch.optim.SGD([W, b], lr=0.1)
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = cross_entropy(net(X), y).mean()
        trainer.zero_grad()
        l.backward()
        trainer.step()
    print(f'epoch {epoch+1}: 训练准确率 {evaluate_accuracy(net, train_iter):.3f}, '
          f'测试准确率 {evaluate_accuracy(net, test_iter):.3f}')


# In[ ]:




