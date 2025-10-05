#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.4. 多输入多输出通道"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("6.4. 多输入多输出通道")
print("=" * 50)

# 辅助函数
def d2l_corr2d(X, K):
    """二维互相关运算"""
    h, w = K.shape
    Y = tf.Variable(tf.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j].assign(tf.reduce_sum(X[i: i + h, j: j + w] * K))
    return Y

def corr2d_multi_in(X, K):
    """计算多输入通道的二维互相关运算"""
    return tf.reduce_sum([d2l_corr2d(x, k) for x, k in zip(X, K)], axis=0)

# 6.4.1. 多输入通道
print("\n6.4.1. 多输入通道")

X = tf.constant([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
K = tf.constant([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])

print(f"输入 X 形状 (通道数, 高, 宽): {X.shape}")
print(f"卷积核 K 形状 (通道数, 高, 宽): {K.shape}")
print("\n输出:")
print(corr2d_multi_in(X, K).numpy())

# 6.4.2. 多输出通道
print("\n6.4.2. 多输出通道")

def corr2d_multi_in_out(X, K):
    """计算多输入多输出通道的二维互相关运算"""
    return tf.stack([corr2d_multi_in(X, k) for k in K], 0)

K = tf.stack((K, K + 1, K + 2), 0)
print(f"卷积核 K 形状 (输出通道数, 输入通道数, 高, 宽): {K.shape}")

Y = corr2d_multi_in_out(X, K)
print(f"\n输出 Y 形状 (输出通道数, 高, 宽): {Y.shape}")
print("输出 Y:")
print(Y.numpy())

# 6.4.3. 1×1 卷积层
print("\n6.4.3. 1×1 卷积层")

def corr2d_multi_in_out_1x1(X, K):
    """1×1卷积的等价实现"""
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = tf.reshape(X, (c_i, h * w))
    K = tf.reshape(K, (c_o, c_i))
    Y = tf.matmul(K, X)
    return tf.reshape(Y, (c_o, h, w))

X_test = tf.random.normal((3, 3, 3), 0, 1)
K_test = tf.random.normal((2, 3, 1, 1), 0, 1)

Y1 = corr2d_multi_in_out_1x1(X_test, K_test)
Y2 = corr2d_multi_in_out(X_test, K_test)

print(f"两种方法结果是否相同: {float(tf.reduce_sum(tf.abs(Y1 - Y2))) < 1e-6}")
print("1×1卷积本质上是每个像素位置应用全连接层！")

# 6.4.4. 实际应用
print("\n6.4.4. 实际应用示例")

X_app = tf.random.uniform((1, 32, 32, 3))

# 3个输入通道 -> 64个输出通道
conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same')
Y1 = conv1(X_app)
print(f"第一层: {X_app.shape} -> {Y1.shape}")

# 64个输入通道 -> 128个输出通道
conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, padding='same')
Y2 = conv2(Y1)
print(f"第二层: {Y1.shape} -> {Y2.shape}")

# 使用1×1卷积调整通道数：128 -> 256
conv3 = tf.keras.layers.Conv2D(256, kernel_size=1)
Y3 = conv3(Y2)
print(f"1×1卷积: {Y2.shape} -> {Y3.shape}")

print("\n✅ 6.4 多输入多输出通道代码运行完成!")
