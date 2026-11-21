#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch

A = torch.arange(20).reshape(5, 4)
B = A.T                      # 转置：行列互换
print(A)
print(B)
print(A.T == B)              # 验证：转置的转置回到自己


# In[3]:


B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
print(B == B.T)              # 全 True → B 是对称矩阵


# In[6]:


A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = torch.arange(12, dtype=torch.float32).reshape(4, 3)
C = torch.mm(A, B)           # 矩阵乘法，等价写法 A @ B
print(C.shape)               # torch.Size([5, 3])

# 手工验证一个元素：C[0,0] = A 第0行 · B 第0列
a_row = A[0, :]              # [0, 1, 2, 3]
b_col = B[:, 0]              # [0, 4, 8]
print(sum(a_row * b_col))    # 0*0 + 1*4 + 2*8 = 20
print(C[0, 0])               # tensor(20.) —— 对上了！


# In[8]:


I = torch.eye(4) 
print(I)


# In[9]:


u = torch.tensor([3.0,-4.0])
print(torch.norm(u))
print(torch.abs(u).sum())


# In[10]:


A = torch.arange(20,dtype=torch.float32).reshape(5,4)
print(A.sum())
print(A.sum(axis=0))
print(A.sum(axis=1))
print(A.mean(axis=0))


# In[ ]:




