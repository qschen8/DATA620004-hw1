import numpy as np
from utils import *

class ThreeLayerNet:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu'):
        self.W1 = np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2./input_size)
        self.b1 = np.zeros(hidden_sizes[0])
        self.W2 = np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(2./hidden_sizes[0])
        self.b2 = np.zeros(hidden_sizes[1])
        self.W3 = np.random.randn(hidden_sizes[1], output_size) * np.sqrt(2./hidden_sizes[1])
        self.b3 = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, X):
        self.z1 = X.dot(self.W1) + self.b1
        self.a1 = relu(self.z1) if self.activation == 'relu' else sigmoid(self.z1)
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = relu(self.z2) if self.activation == 'relu' else sigmoid(self.z2)
        scores = self.a2.dot(self.W3) + self.b3
        return scores
    
    def backward(self, X, y, reg):
        m = X.shape[0]
        grad = {}
        
        # 交叉熵梯度
        scores = self.forward(X)
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        dscores = (probs - one_hot_encode(y, 10)) / m
        
        # 第三层梯度
        grad['W3'] = self.a2.T.dot(dscores) + reg * self.W3
        grad['b3'] = np.sum(dscores, axis=0)
        
        # 第二层梯度
        da2 = dscores.dot(self.W3.T)
        da2[self.z2 <= 0] = 0 if self.activation == 'relu' else da2 * sigmoid_deriv(self.z2)
        grad['W2'] = self.a1.T.dot(da2) + reg * self.W2
        grad['b2'] = np.sum(da2, axis=0)
        
        # 第一层梯度
        da1 = da2.dot(self.W2.T)
        da1[self.z1 <= 0] = 0 if self.activation == 'relu' else da1 * sigmoid_deriv(self.z1)
        grad['W1'] = X.T.dot(da1) + reg * self.W1
        grad['b1'] = np.sum(da1, axis=0)
        
        return grad
