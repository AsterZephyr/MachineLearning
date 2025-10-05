#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.3. 填充和步幅"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("6.3. 填充和步幅")
print("=" * 50)

# 定义辅助函数
def comp_conv2d(conv2d, X):
    X = tf.reshape(X, (1, ) + X.shape + (1, ))
    Y = conv2d(X)
    return tf.reshape(Y, Y.shape[1:3])

# 6.3.1. 填充
print("\n6.3.1. 填充")

conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same')
X = tf.random.uniform(shape=(8, 8))
output = comp_conv2d(conv2d, X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print("填充后，输出形状与输入形状相同！")

# 不同大小的卷积核
print("\n使用5×3卷积核:")
conv2d = tf.keras.layers.Conv2D(1, kernel_size=(5, 3), padding='same')
output = comp_conv2d(conv2d, X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")

# 6.3.2. 步幅
print("\n6.3.2. 步幅")

conv2d = tf.keras.layers.Conv2D(1, kernel_size=3, padding='same', strides=2)
output = comp_conv2d(conv2d, X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")
print("步幅为2时，输出的高度和宽度都减半！")

# 复杂例子
print("\n复杂配置:")
conv2d = tf.keras.layers.Conv2D(
    1, kernel_size=(3, 5), padding='valid', strides=(3, 4))
output = comp_conv2d(conv2d, X)
print(f"输入形状: {X.shape}")
print(f"输出形状: {output.shape}")

# 6.3.3. 实际应用
print("\n6.3.3. 实际应用示例")

X1 = tf.random.uniform((1, 32, 32, 3))

# 配置1: 保持尺寸
conv1 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', strides=1)
Y1 = conv1(X1)
print(f"配置1 - 保持尺寸: {X1.shape} -> {Y1.shape}")

# 配置2: 下采样
conv2 = tf.keras.layers.Conv2D(64, kernel_size=3, padding='same', strides=2)
Y2 = conv2(X1)
print(f"配置2 - 下采样2倍: {X1.shape} -> {Y2.shape}")

# 配置3: 大核下采样
conv3 = tf.keras.layers.Conv2D(128, kernel_size=5, padding='same', strides=2)
Y3 = conv3(X1)
print(f"配置3 - 下采样2倍（大核）: {X1.shape} -> {Y3.shape}")

print("\n✅ 6.3 填充和步幅代码运行完成!")
