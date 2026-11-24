#!/usr/bin/env python
# coding: utf-8

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')
import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声（书里的函数）"""
    X = torch.normal(0, 1, (num_examples, len(w)))   # 特征矩阵：1000×2，标准正态
    y = torch.matmul(X, w) + b                       # 真实输出（不加噪声的版本）
    y += torch.normal(0, 0.01, y.shape)              # 加噪声：模拟现实数据的不完美
    return X, y.reshape((-1, 1))                     # y 变成 1000×1 的列

true_w = torch.tensor([2, -3.4])     # 藏起来的"真值"
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

print(features[0], labels[0])        # 看一眼第一个样本


# In[21]:


d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.show()      


# In[22]:


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)                      # 打乱顺序（随机性的来源）
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i: i + batch_size])
        yield features[batch_indices], labels[batch_indices]   # 吐出一小撮

batch_size = 10

# 试运行：看第一个批次长什么样
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)      # X 是 10×2，y 是 10×1 —— 一个 batch
    break


# In[23]:


w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)   # 随机初始化！
b = torch.zeros(1, requires_grad=True)


# In[24]:


def linreg(X, w, b):
    """线性回归模型：ŷ = Xw + b"""
    return torch.matmul(X, w) + b


# In[25]:


def squared_loss(y_hat, y):
    """均方损失（书里版本：不取均值，后面除）"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# In[26]:


def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():                # 更新参数时不要追踪梯度
        for param in params:
            param -= lr * param.grad / batch_size   # 沿负梯度走一小步
            param.grad.zero_()                       # 清零！否则梯度会累加


# In[28]:


lr = 0.03                  # 学习率
num_epochs = 3             # 全部数据过 3 遍
net = linreg               # 模型（起别名，简洁实现时一换即可）
loss = squared_loss        # 损失

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)     # ① 预测 + ② 算损失（100×...不对，10 个损失）
        l.sum().backward()            # ③ 反向传播：算出 w.grad 和 b.grad
        sgd([w, b], lr, batch_size)   # ④ 用梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

# ---- 训练完成，验收：学到的参数 vs 真值 ----
print(f'w 的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b 的估计误差: {true_b - b}')


# In[ ]:




