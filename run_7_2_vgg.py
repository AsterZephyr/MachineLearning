#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.2. VGG - 使用块的网络"""

import tensorflow as tf
import numpy as np

print("=" * 70)
print("7.2. VGG - 使用块的网络")
print("=" * 70)

print("""
VGG网络由牛津大学的Visual Geometry Group提出。

核心思想：
1. 使用可重复的卷积块构建网络
2. 所有卷积核都是3×3
3. 通过堆叠多个3×3卷积代替大卷积核
4. 每个VGG块包含：多个3×3卷积 + ReLU + 一个2×2最大池化

优势：
- 模块化设计，易于扩展
- 多个小卷积核比单个大卷积核参数更少
- 3×3卷积的感受野足够，且增加了网络深度
""")

def vgg_block(num_convs, num_channels):
    """
    VGG块

    参数:
        num_convs: 卷积层数量
        num_channels: 输出通道数
    """
    block = tf.keras.models.Sequential()
    for _ in range(num_convs):
        block.add(tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same', activation='relu'))
    block.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    return block

# VGG-11架构（简化版）
def VGG():
    """
    VGG-11网络

    架构: [(1个3×3卷积, 64通道),
           (1个3×3卷积, 128通道),
           (2个3×3卷积, 256通道),
           (2个3×3卷积, 512通道),
           (2个3×3卷积, 512通道)] + 全连接层
    """
    # 卷积块配置：(卷积层数量, 输出通道数)
    conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))

    model = tf.keras.models.Sequential()

    # 添加卷积块
    for (num_convs, num_channels) in conv_arch:
        model.add(vgg_block(num_convs, num_channels))

    # 添加全连接层
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(4096, activation='relu'))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(10))

    return model

# 创建模型
print("\n构建VGG-11模型...")
net = VGG()

# 查看数据流
X = tf.random.uniform((1, 224, 224, 1))
net(X)  # 初始化
print("\n数据流动过程:")
print(f"输入: {X.shape}")

# 追踪每个块的输出
X_test = X
block_num = 0
for layer in net.layers:
    X_test = layer(X_test)
    if isinstance(layer, tf.keras.models.Sequential):
        block_num += 1
        print(f"VGG块{block_num}: {X_test.shape}")
    elif isinstance(layer, tf.keras.layers.Flatten):
        print(f"展平: {X_test.shape}")
    elif isinstance(layer, tf.keras.layers.Dense):
        print(f"全连接: {X_test.shape}")

# 模型总结
print("\n" + "=" * 70)
print("VGG模型架构总结")
print("=" * 70)
net.summary()

print("\n" + "=" * 70)
print("VGG的关键设计理念")
print("=" * 70)
print("""
1. 【3×3卷积核的优势】
   两个3×3卷积 = 一个5×5卷积的感受野
   但参数量：2×(3×3) = 18 < 5×5 = 25
   而且增加了一个非线性激活函数（ReLU）

2. 【模块化设计】
   每个VGG块都是独立的模块
   可以灵活组合成不同深度的网络：
   - VGG-11: 8个卷积层 + 3个全连接层
   - VGG-16: 13个卷积层 + 3个全连接层
   - VGG-19: 16个卷积层 + 3个全连接层

3. 【通道数规律】
   通道数依次加倍：64 → 128 → 256 → 512 → 512
   这样可以逐步提取更抽象的特征

4. 【空间维度规律】
   每个池化层将尺寸减半
   224 → 112 → 56 → 28 → 14 → 7
   最后通过全连接层进行分类
""")

print("\n✅ 7.2 VGG代码运行完成!")
