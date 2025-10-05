#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.1. AlexNet - 深度卷积神经网络"""

import tensorflow as tf
import numpy as np

print("=" * 70)
print("7.1. AlexNet - 深度卷积神经网络")
print("=" * 70)

print("""
AlexNet是2012年ImageNet挑战赛的冠军，标志着深度学习时代的开始。

关键创新：
1. 更深的网络（8层：5个卷积层 + 3个全连接层）
2. 使用ReLU激活函数（而非Sigmoid）
3. 使用Dropout防止过拟合
4. 数据增强
5. GPU加速训练

与LeNet的主要区别：
- 更大的卷积核（11×11, 5×5, 3×3）
- 更多的输出通道（96, 256, 384...）
- ReLU激活函数
- Dropout正则化
""")

# 构建AlexNet（简化版，适配Fashion-MNIST）
def AlexNet():
    return tf.keras.models.Sequential([
        # 第1层：卷积层，大卷积核11×11，步幅4
        tf.keras.layers.Conv2D(
            filters=96, kernel_size=11, strides=4,
            activation='relu', input_shape=(224, 224, 1)),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 第2层：卷积层，5×5
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=5, padding='same',
            activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 第3层：卷积层，3×3
        tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, padding='same',
            activation='relu'),

        # 第4层：卷积层，3×3
        tf.keras.layers.Conv2D(
            filters=384, kernel_size=3, padding='same',
            activation='relu'),

        # 第5层：卷积层，3×3
        tf.keras.layers.Conv2D(
            filters=256, kernel_size=3, padding='same',
            activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # 展平
        tf.keras.layers.Flatten(),

        # 第6层：全连接层，使用Dropout
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # 第7层：全连接层，使用Dropout
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dropout(0.5),

        # 第8层：输出层
        tf.keras.layers.Dense(10)
    ])

# 创建模型
print("\n构建AlexNet模型...")
net = AlexNet()

# 查看数据流
X = tf.random.uniform((1, 224, 224, 1))
print("\n数据流动过程:")
print(f"输入形状: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    print(f"第{i+1}层 ({layer.__class__.__name__:15s}): {X.shape}")

# 模型总结
print("\n" + "=" * 70)
print("AlexNet模型架构总结")
print("=" * 70)
net.summary()

print("\n关键参数统计:")
total_params = net.count_params()
print(f"总参数量: {total_params:,}")
print(f"相比LeNet的61,706个参数，增加了 {total_params/61706:.1f} 倍")

print("\n✅ 7.1 AlexNet代码运行完成!")
print("\n注意事项:")
print("- AlexNet原本设计用于ImageNet（1000类），这里简化为10类")
print("- 原始输入为224×224×3（RGB），这里简化为224×224×1（灰度）")
print("- 由于参数量大，完整训练需要较长时间和GPU支持")
