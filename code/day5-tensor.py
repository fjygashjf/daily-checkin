#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch

x = torch.arange(12)
print(x)           # tensor([ 0,  1, ..., 11])
print(x.shape)     # 形状：每维多长
print(x.numel())   # 元素总数


# In[2]:


X = x.reshape(3, 4)
print(X)
# 观察：按行填充；3×4=12 元素总数没变 —— reshape 只改摆放不增删元素
print(x.reshape(-1, 4))   # -1 = 这维让机器自己推（自动 3）


# In[3]:


print(torch.zeros(2, 3, 4))   # 全 0 —— 感知器 w1,w2=0.0 的"批量版"
print(torch.ones(2, 3, 4))    # 全 1
print(torch.randn(3, 4))      # 标准正态随机数 —— 真实网络权重初始化来源


# In[4]:


y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(y)


# In[5]:


a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([10.0, 20.0, 30.0])
print(a + b)    # [11., 22., 33.]   对应位置相加
print(a * b)    # [10., 40., 90.]   ★ 逐元素乘，不是矩阵乘！


# In[6]:


print(a + 100)   # [101., 102., 103.] 标量自动扩展后参与运算


# In[7]:


print(X[1, 2])     # 第 1 行第 2 列
print(X[:, 1])     # 所有行的第 1 列
X[0, 0] = 99       # 张量元素可修改
print(X[0])


# In[ ]:




