#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)   # ← 书里把昨天的造数据函数也封装了


# In[9]:


def load_array(data_arrays, batch_size, is_train=True):
    """构造 PyTorch 数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)        # 把 X、y 打包成数据集
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 试读一个批次
print(next(iter(data_iter)))


# In[10]:


from torch import nn

net = nn.Sequential(nn.Linear(2, 1))    # ← 惊人的短


# In[11]:


print(net[0].weight)      # 参数初始值（随机）
print(net[0].bias)


# In[12]:


net[0].weight.data.normal_(0, 0.01)   # 权重：正态(0, 0.01) —— 和昨天一样
net[0].bias.data.fill_(0)             # 偏置：0


# In[13]:


loss = nn.MSELoss()                          # 平方损失（自动取均值！）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)   # SGD 优化器


# In[14]:


num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        l = loss(net(X), y)          # ① 预测+算损失（注意：b 已藏在 net 里！）
        trainer.zero_grad()          # ② 清零梯度（=昨天的 param.grad.zero_()）
        l.backward()                 # ③ 反向传播（不变——autograd 不可替代）
        trainer.step()               # ④ 更新参数（=昨天的 sgd 函数调用）
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

