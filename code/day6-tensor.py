
import pandas as pd

# 造一张有缺失值的表（书上例子：房屋数据）
data = pd.DataFrame({
    'NumRooms': [2, 4, None, 3],      # None = 缺失值，读进来显示 NaN
    'Alley':    ['Pave', None, None, None],
    'Price':    [127500, 106000, 178100, 140000],
})
print(data)

# 思路一：插补——用该列平均值填充缺失的房间数
inputs = data.iloc[:, 0:2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))
print(inputs)

# 思路二（对照理解）：把缺失行整行删掉
print(data.dropna())

# 转张量：pandas 的"哑变量"处理（Alley 列文字变 0/1 列）后转 tensor
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)
import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
print(X)


# In[ ]:




