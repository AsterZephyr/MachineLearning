#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.5. 池化层"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("6.5. 池化层")
print("=" * 50)

# 6.5.1. 最大池化和平均池化
print("\n6.5.1. 最大池化和平均池化")

def pool2d(X, pool_size, mode='max'):
    """实现池化层的前向传播"""
    p_h, p_w = pool_size
    Y = tf.Variable(tf.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1)))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j].assign(tf.reduce_max(X[i: i + p_h, j: j + p_w]))
            elif mode == 'avg':
                Y[i, j].assign(tf.reduce_mean(X[i: i + p_h, j: j + p_w]))
    return Y

X = tf.constant([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
print("输入 X:")
print(X.numpy())
print("\n最大池化结果（2×2窗口）:")
print(pool2d(X, (2, 2)).numpy())
print("\n平均池化结果（2×2窗口）:")
print(pool2d(X, (2, 2), 'avg').numpy())

# 6.5.2. 填充和步幅
print("\n6.5.2. 填充和步幅")

X = tf.reshape(tf.range(16, dtype=tf.float32), (1, 4, 4, 1))
print("输入 X:")
print(X[0, :, :, 0].numpy())

# 默认步幅
pool2d_layer = tf.keras.layers.MaxPool2D(pool_size=[3, 3])
result = pool2d_layer(X)
print(f"\n3×3池化，默认步幅=3: {X.shape} -> {result.shape}")
print(result[0, :, :, 0].numpy())

# 手动设置填充和步幅
pool2d_layer = tf.keras.layers.MaxPool2D(
    pool_size=[3, 3], padding='same', strides=2)
result = pool2d_layer(X)
print(f"\n3×3池化，填充=same，步幅=2: {X.shape} -> {result.shape}")
print(result[0, :, :, 0].numpy())

# 矩形池化窗口
pool2d_layer = tf.keras.layers.MaxPool2D(
    pool_size=[2, 3], padding='same', strides=(2, 3))
result = pool2d_layer(X)
print(f"\n2×3池化，步幅=(2,3): {X.shape} -> {result.shape}")
print(result[0, :, :, 0].numpy())

# 6.5.3. 多个通道
print("\n6.5.3. 多个通道")

X = tf.concat([X, X + 1], 3)
print(f"输入形状（含2个通道）: {X.shape}")

pool2d_layer = tf.keras.layers.MaxPool2D(pool_size=[3, 3], padding='same', strides=2)
result = pool2d_layer(X)
print(f"池化后形状: {result.shape}")
print("\n通道1:")
print(result[0, :, :, 0].numpy())
print("\n通道2:")
print(result[0, :, :, 1].numpy())

# 6.5.4. 实际应用
print("\n6.5.4. 实际应用示例")

X_app = tf.random.uniform((1, 224, 224, 3))
print(f"输入图像: {X_app.shape}")

# 卷积+池化块1
conv1 = tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu')
pool1 = tf.keras.layers.MaxPool2D(2, strides=2)
Y = pool1(conv1(X_app))
print(f"卷积+池化后: {Y.shape} (尺寸减半)")

# 卷积+池化块2
conv2 = tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu')
pool2 = tf.keras.layers.MaxPool2D(2, strides=2)
Y = pool2(conv2(Y))
print(f"再次卷积+池化: {Y.shape} (再次减半)")

# 全局平均池化
global_pool = tf.keras.layers.GlobalAveragePooling2D()
Y = global_pool(Y)
print(f"全局平均池化: {Y.shape} (空间维度消失)")

print("\n✅ 6.5 池化层代码运行完成!")
