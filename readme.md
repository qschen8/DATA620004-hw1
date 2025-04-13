数据预处理（依据）
 
•	数据集划分：CIFAR-10包含50k训练+10k测试图像。将50k训练集划分为45k训练+5k验证集。
•	归一化处理：将像素值从[0,255]线性映射到[-1,1]（公式：(X/127.5)-1）。
•	数据加载器：实现mini-batch处理，batch_size=64，使用numpy的np.memmap高效加载二进制数据。
•	2. 模型架构（三层神经网络）
'''python
class ThreeLayerNet:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        # 权重初始化（He初始化）
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2./input_size)
        self.b1 = np.zeros(hidden_sizes[0])
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2./hidden_sizes[0])
        self.b2 = np.zeros(hidden_sizes[1])
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2./hidden_sizes[1])
        self.b3 = np.zeros(output_size)
        self.activation = activation  # 支持relu/sigmoid
'''


4. 训练过程（依据）
•	优化器：SGD with momentum（动量系数0.9），学习率指数衰减（每epoch衰减5%）。
•	正则化：L2正则化项加入损失计算，公式：loss = data_loss + 0.5*reg*(np.sum(W1**2)+np.sum(W2**2)+np.sum(W3**2))。
•	早停机制：当验证集准确率连续3个epoch不提升时终止训练，保存最佳模型。

