#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import random

# 蒙特卡洛实验：用随机数"掷骰子"，观察大数定律
rolls = torch.randint(1, 7, (10000,))     # 掷 1 万次骰子
for n in [10, 100, 1000, 10000]:
    est = rolls[:n].float().mean()        # 前 n 次的平均点数
    print(n, est.item())


# In[2]:


print(torch.randn.__doc__)   


# In[ ]:




