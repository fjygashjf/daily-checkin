# -*- coding: utf-8 -*-
"""第 3 天打卡：手写感知器，学会"与门"
纯 Python 实现，不依赖任何第三方库
"""

# ---------- 1. 训练数据：与门 ----------
# 每个样本是 (x1, x2, 正确答案y)
data = [
    (0, 0, 0),
    (0, 1, 0),
    (1, 0, 0),
    (1, 1, 1),
]

# ---------- 2. 初始化 ----------
w1, w2 = 0.0, 0.0   # 权重：一开始是"瞎猜"的
b = 0.0             # 偏置：相当于"默认倾向"
lr = 0.1            # 学习率：每次纠错调多大幅度
epochs = 20         # 把全部样本过多少轮

# ---------- 3. 感知器的"判断"函数 ----------
def predict(x1, x2):
    """加权求和，过阈值，输出 0 或 1"""
    z = w1 * x1 + w2 * x2 + b
    return 1 if z > 0 else 0

# ---------- 4. 训练循环 ----------
for epoch in range(epochs):
    errors = 0
    for x1, x2, y in data:
        pred = predict(x1, x2)   # 先让它猜
        if pred != y:            # 猜错了
            # 感知器学习规则：权重往正确方向挪一小步
            w1 += lr * (y - pred) * x1
            w2 += lr * (y - pred) * x2
            b  += lr * (y - pred)
            errors += 1
    print(f"第{epoch + 1}轮: 错{errors}次, w1={w1:.2f}, w2={w2:.2f}, b={b:.2f}")
    if errors == 0:              # 全部答对，提前毕业
        print(">>> 训练完成，感知器学会了与门！")
        break

# ---------- 5. 验收考试 ----------
print("\n最终验证：")
for x1, x2, y in data:
    print(f"输入({x1},{x2}) -> 预测{predict(x1, x2)}, 正确答案{y}, {'对' if predict(x1, x2) == y else '错'}")
