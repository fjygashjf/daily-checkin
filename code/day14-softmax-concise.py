#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)


# In[2]:


net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)     # 初始化：正态(0, 0.01)，和手写版一致


# In[3]:


loss = nn.CrossEntropyLoss(reduction='mean')    # ← 今天的灵魂，见下方讲解
trainer = torch.optim.SGD(net.parameters(), lr=0.1)


# In[4]:


num_epochs = 3
for epoch in range(num_epochs):
    for X, y in train_iter:
        l = loss(net(X), y)          # logits 直接进损失
        trainer.zero_grad()
        l.backward()
        trainer.step()
    # 评估
    net.eval()
    with torch.no_grad():
        def acc(data_iter):
            c, t = 0, 0
            for X, y in data_iter:
                c += (net(X).argmax(axis=1).type(y.dtype) == y).sum().item()
                t += y.numel()
            return c / t
    print(f'epoch {epoch+1}: 训练准确率 {acc(train_iter):.3f}, 测试准确率 {acc(test_iter):.3f}')

