import datetime
import numpy as np
import os
from test import evaluate
from utils import *
from data_loader import load_cifar10
from model import ThreeLayerNet
from utils import plot_training
from tqdm import tqdm

def train(model, X_train, y_train, X_val, y_val, 
          lr=1e-3, reg=1e-4, epochs=100, batch_size=256,
          lr_decay=0.95, patience=3):
    
    best_acc = -1
    history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
    velocity = {k: np.zeros_like(v) for k, v in model.__dict__.items() if k.startswith('W') or k.startswith('b')}
    momentum = 0.9
    
    epoch_bar = tqdm(range(epochs), desc='Training')

    for epoch in epoch_bar:
        # 学习率衰减
        lr *= lr_decay
        
        # 随机打乱数据
        permutation = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[permutation]
        y_shuffled = y_train[permutation]

        batch_iter = range(0, X_train.shape[0], batch_size)
        batch_bar = tqdm(batch_iter, desc=f"Epoch {epoch+1}", leave=False)

        epoch_train_loss = 0.0
        num_batches = 0
        
        for i in batch_bar:
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]
            
            # 前向传播
            scores = model.forward(X_batch)
            
            # 计算损失
            data_loss = cross_entropy_loss(scores, y_batch)
            reg_loss = 0.5 * reg * (np.sum(model.W1**2) + np.sum(model.W2**2) + np.sum(model.W3**2))
            total_loss = data_loss + reg_loss
            epoch_train_loss += total_loss
            num_batches += 1

            # 反向传播
            grads = model.backward(X_batch, y_batch, reg)
            
            # 动量更新参数
            for param in model.__dict__:
                if param in grads:
                    velocity[param] = momentum * velocity[param] - lr * grads[param]
                    model.__dict__[param] += velocity[param]
            
            batch_bar.set_postfix({
                    'loss': f'{total_loss:.4f}',
                    'lr': f'{lr:.4f}'
                    })
        
        avg_train_loss = epoch_train_loss / num_batches
        history['train_loss'].append(avg_train_loss)
        
        # 验证集评估
        val_acc = evaluate(model, X_val, y_val)
        val_loss = cross_entropy_loss(model.forward(X_val), y_val) + 0.5*reg*(np.sum(model.W1**2)+np.sum(model.W2**2)+np.sum(model.W3**2))
        
        # 记录历史
        history['val_acc'].append(val_acc)
        history['val_loss'].append(val_loss)

        epoch_bar.set_postfix({
            "train_loss": f"{avg_train_loss:.4f}",  
            "val_acc": f"{val_acc:.2%}",
            "best_acc": f"{best_acc:.2%}"
        })        
        
        # 早停和保存最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            np.savez('best_model.npz', **model.__dict__)
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
            
    return history


if __name__ == '__main__':
    print('Training...')
    time = datetime.datetime.now().strftime("%Y-%m-%d")
    # 加载数据
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = load_cifar10('data/cifar-10-batches-py')
    
    # 初始化模型
    model = ThreeLayerNet(input_size=3072, hidden_sizes=(2048, 1024), output_size=10)
    save_path = f'train_curve.png'
    
    # Traing
    history = train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lr=3e-2,
        reg=5e-4,
        epochs=100,
        batch_size=256
    )
    
    # 绘制训练曲线
    plot_training(history, save_path)
