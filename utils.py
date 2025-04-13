import numpy as np
import matplotlib.pyplot as plt

def one_hot_encode(y, num_classes):
    return np.eye(num_classes)[y]

def cross_entropy_loss(scores, y):
    m = y.shape[0]
    log_probs = -np.log(softmax(scores)[range(m), y])
    return np.sum(log_probs) / m

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))

def plot_training(history, save_path):
    plt.figure(figsize=(12,4))
    plt.subplot(121)
    plt.plot(history['train_loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title('Loss curve')
    plt.legend()
    
    plt.subplot(122)
    plt.plot(history['val_acc'])
    plt.title('Val accuracy')
    plt.show()
    plt.savefig(save_path)
    plt.close()
