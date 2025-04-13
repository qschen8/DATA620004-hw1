import numpy as np
from model import ThreeLayerNet
from data_loader import load_cifar10
import numpy as np

def evaluate(model, X, y):
    scores = model.forward(X)
    preds = np.argmax(scores, axis=1)
    return np.mean(preds == y)

def load_model(model_class, filename, input_size=3072, hidden_sizes=(512,256), output_size=10):
    model = model_class(input_size, hidden_sizes, output_size)
    params = np.load(filename)
    for name in params.files:
        model.__dict__[name] = params[name]
    return model


if __name__ == '__main__':
    best_model_path = 'best_model.npz'
    # 加载数据
    _, _, (X_test, y_test) = load_cifar10('data/cifar-10-batches-py')
    
    # 加载最佳模型
    def load_best_model():
        model = ThreeLayerNet(3072, (2048, 1024), 10)
        params = np.load(best_model_path, allow_pickle=True)
        for key in params:
            model.__dict__[key] = params[key]
        return model
    
    model = load_best_model()
    acc = evaluate(model, X_test, y_test)
    print(f'Test Accuracy: {acc*100:.2f}%')

    # 写入结果
    with open('test_acc.txt', 'w') as f:
        f.write(f"Test Accuracy: {acc*100:.2f}%")

