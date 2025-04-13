import numpy as np
import pickle
import os

def load_cifar10(data_dir):
    """加载CIFAR-10数据集"""
    train_data, train_labels = [], []
    for i in range(1,6):
        with open(os.path.join(data_dir, f'data_batch_{i}'), 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
            train_data.append(batch['data'])
            train_labels.append(batch['labels'])
    
    X_train = np.concatenate(train_data).astype(np.float32)
    y_train = np.concatenate(train_labels)
    
    with open(os.path.join(data_dir, 'test_batch'), 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
        X_test = batch['data'].astype(np.float32)
        y_test = np.array(batch['labels'])
    
    # 归一化到[-1,1]
    X_train = (X_train / 127.5) - 1
    X_test = (X_test / 127.5) - 1
    
    # 划分验证集
    val_mask = np.zeros(X_train.shape[0], dtype=bool)
    val_mask[:5000] = True  # 前5000作为验证集
    X_val, y_val = X_train[val_mask], y_train[val_mask]
    X_train, y_train = X_train[~val_mask], y_train[~val_mask]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
