import torch            #导入torch库就是导入了pytorch库
import torch.nn as nn   #导入nn模块就是导入了pytorch的神经网络模块

# 1. 造数据：100个样本，每个2个数字，标签是"两数之和是否大于1"
X = torch.rand(100, 2)  #有随机生成的100行（样本），每行有2个数字（特征）///训练数据
y = (X[:, 0] + X[:, 1] > 1).float().unsqueeze(1)
#X[:, 0]取X的第0列，X[:, 1]取X的第1列，得到布尔值,然后float（）转为浮点数，后通过unsqueeze（1）升维度

# 2. 搭神经网络：2层
model = nn.Sequential(
    nn.Linear(2, 16),   #将2维特征转换为16神经元，通过线性运算
    nn.Sigmoid(),                            # 把16个神经元的值压缩到0-1之间
    nn.Linear(16, 1),   #将16个神经元转换为1个神经元，通过线性运算
    nn.Sigmoid()                             # 把1个神经元的值压缩到0-1之间
)

# 3. 损失函数 + 优化器
criterion = nn.BCELoss()                      #计算预测和真实标签之间的差值（损失函数）对应sigmoid函数
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
#SGD是随机梯度下降，model.parameters()调整合适的权重来修改得到最小损失，lr是学习率，lr太大不好收敛，lr太小训练太慢

# 4. 训练1000轮
for epoch in range(5000):                       #循环5000轮
    pred = model(X)                             # 前向传播：算预测
    loss = criterion(pred, y)                   # 算损失：错了多少
    optimizer.zero_grad()                       # 清空梯度
    loss.backward()                             # 反向传播：算梯度
    optimizer.step()                            # 梯度下降：更新权重

    if (epoch + 1) % 1000 == 0:                  #每1000轮输出一次损失
        print(f"第 {epoch + 1} 轮，损失: {loss.item():.4f}")  #输出损失

# 5. 测试
print("\n测试：")
test = torch.tensor([[0.2, 0.3], [0.8, 0.9]])    #测试数据///测试数据
with torch.no_grad():                            #关闭梯度计算
    pred = model(test)                           #前向传播：算预测
    print(f"[0.2, 0.3] 和=0.5<1 → 预测: {pred[0].item():.4f}（应接近0）")
    print(f"[0.8, 0.9] 和=1.7>1 → 预测: {pred[1].item():.4f}（应接近1）")
