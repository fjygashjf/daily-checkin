# 2026-11-28 周六 · softmax 简洁实现

## 今日完成
- 用 nn.Flatten + nn.Linear + nn.CrossEntropyLoss + SGD 重写 softmax 训练
- 两版准确率对照：手写版 ~0.8x，框架版 ~0.8x —— 结果一致，手写实现验证通过

## 今天的核心认知
1. softmax 从"模型的一部分"移进了"损失内部"：网络只输出 logits，
   CrossEntropyLoss 内部做 log 域稳定计算（不怕 exp 溢出）
2. CrossEntropyLoss 吃 logits，MSELoss 吃预测值——
   每换损失都要核对"它期待什么输入"
3. 手写版的调试经历兑现：知道损失内部藏着什么，才不会喂错输入

## 手写↔框架对照
- 展平→Flatten / 参数→自动创建 / softmax+交叉熵→合进一个Loss /
  参数收集→net.parameters() / 训练骨架→不变

## 明日计划
- 周日复盘日：《感知机》一节（轻量）+ 第 3 章收尾检查
