from itertools import product
from utils import *
from model import *
from train import *

def hyperparameter_search():
    param_grid = {
        'hidden_sizes': [(1024,512), (2024, 1024)],
        'lr': [8e-2, 5e-2, 3e-2, 1e-2],
        'reg': [1e-4, 1e-5, 5e-5, 5e-4]
    }

    with open('hyperparam_results.txt', 'w') as f:
        f.write("hidden_size,lr,reg,acc\n")

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10('data/cifar-10-batches-py')
    results = []

    for hsize, lr, reg in product(param_grid['hidden_sizes'], 
                                 param_grid['lr'], 
                                 param_grid['reg']):

        model = ThreeLayerNet(3072, hsize, 10)
        history = train(model, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, lr=lr, reg=reg)  # 传入训练数据
        best_acc = max(history['val_acc'])

        with open('hyperparam_results.txt', 'a') as f:
            for res in results:
                f.write(f"hidden_size={res[0]}, lr={res[1]:.0e}, reg={res[2]:.0e}, acc={res[3]*100:.2f}%\n")

        results.append((hsize, lr, reg, best_acc))

    # 输出最佳参数
    best_params = max(results, key=lambda x: x[3])
    with open('hyperparam_results.txt', 'a') as f:
        f.write(f"Best params: hidden_size={best_params[0]}, lr={best_params[1]:.0e}, reg={best_params[2]:.0e}\n")

    print(f"Best params: hidden_size={best_params[0]}, lr={best_params[1]}, reg={best_params[2]}")

    return best_params

if __name__ == '__main__':
    best_params = hyperparameter_search()
