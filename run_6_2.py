#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.2. 图像卷积"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("6.2. 图像卷积")
print("=" * 50)

# 6.2.1. 互相关运算
print("\n6.2.1. 互相关运算")

def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(X[i: i + h, j: j + w] * K))
    return Y

# 验证互相关运算
X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = tf.constant([[0.0, 1.0], [2.0, 3.0]])

print("输入 X:")
print(X.numpy())
print("\n卷积核 K:")
print(K.numpy())
print("\n输出 Y:")
print(corr2d(X, K).numpy())

# 6.2.4. 边缘检测
print("\n6.2.4. 边缘检测")

# 构造一个6×8像素的图像
X = tf.Variable(tf.ones((6, 8)))
X[:, 2:6].assign(tf.zeros((6, 4)))

print("原始图像:")
print(X.numpy().astype(int))

# 构造一个高度为1、宽度为2的卷积核
K = tf.constant([[1.0, -1.0]])

# 执行互相关运算
Y = corr2d(X, K)
print("\n边缘检测结果:")
print(Y.numpy())
print("\n可以看到，1表示从白到黑的边缘，-1表示从黑到白的边缘")

# 6.2.5. 学习卷积核
print("\n6.2.5. 学习卷积核")

conv2d = tf.keras.layers.Conv2D(1, (1, 2), use_bias=False)

X_reshaped = tf.reshape(X, (1, 6, 8, 1))
Y_reshaped = tf.reshape(Y, (1, 6, 7, 1))

lr = 3e-2

Y_hat = conv2d(X_reshaped)
for i in range(10):
    with tf.GradientTape() as g:
        Y_hat = conv2d(X_reshaped)
        l = (abs(Y_hat - Y_reshaped)) ** 2

    grads = g.gradient(l, conv2d.trainable_variables)
    weights = conv2d.get_weights()
    weights[0] = weights[0] - lr * grads[0]
    conv2d.set_weights(weights)

    if (i + 1) % 2 == 0:
        print(f'epoch {i + 1}, loss {tf.reduce_sum(l):.3f}')

print("\n学习到的卷积核:")
print(tf.reshape(conv2d.weights[0], (1, 2)).numpy())
print("与真实卷积核 [1, -1] 非常接近！")

print("\n✅ 6.2 图像卷积代码运行完成!")
