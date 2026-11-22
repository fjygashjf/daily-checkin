#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
from d2l import torch as d2l

def f(x): return 3 * x ** 2 - 4 * x   # 书里的例子 f(x) = 3x² - 4x
def num_grad(x): return (f(x + 0.000001) - f(x)) / 0.000001  # 数值近似

print(f(1), num_grad(1))   


# In[3]:


x = np.arange(0, 3, 0.1)
d2l.plot(x, [3 * x ** 2 - 4 * x + 0, 6 * x - 8], 'x', 'f(x)', legend=['f(x)', '切线'])


# In[4]:


import torch

x = torch.arange(4.0, requires_grad=True)   # 追踪 x
print(x)                                    # tensor([0., 1., 2., 3.])

y = 2 * torch.dot(x, x)                     # y = 2·x·x（向量点积再乘2）
y.backward()                                # 反向：自动求 y 对 x 的梯度
print(x.grad)                               # tensor([ 0., 4., 8., 12.])


# In[5]:


x.grad.zero_()                              # 梯度默认会累加，先清零
y = x.sum()
y.backward()
print(x.grad)                               # 全 1（因为 sum 对每个分量的导数都是 1）


# In[ ]:




