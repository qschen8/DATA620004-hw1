import numpy as np
import matplotlib.pyplot as plt
from model import ThreeLayerNet

# 加载模型
best_model_path = 'best_model.npz'
model = ThreeLayerNet(3072, (2048, 1024), 10)
params = np.load(best_model_path)
model.W1 = params['W1']

# 可视化第一层权重
plt.figure(figsize=(15,6))
for i in range(32):
    plt.subplot(4, 8, i+1)
    w = model.W1[:,i].reshape(32,32,3)
    w = (w - w.min()) / (w.max() - w.min())
    plt.imshow(w)
    plt.axis('off')
plt.savefig('weight_visualization.png')
plt.close()
