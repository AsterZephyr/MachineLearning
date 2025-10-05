#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.5. 批量归一化 (Batch Normalization)"""

import tensorflow as tf
import numpy as np

print("=" * 70)
print("7.5. 批量归一化 (Batch Normalization)")
print("=" * 70)

print("""
批量归一化的核心思想：
1. 在训练时，对每个小批量的数据进行标准化
2. 使用可学习的参数进行缩放和平移
3. 加速训练，提高模型性能

数学公式：
BN(x) = γ * (x - μ) / σ + β

其中：
- μ: 批量均值
- σ: 批量标准差
- γ: 可学习的缩放参数
- β: 可学习的平移参数

为什么有效？
1. 减少内部协变量偏移
2. 允许使用更大的学习率
3. 降低对初始化的敏感度
4. 起到正则化作用
""")

# 不使用批量归一化的LeNet
def LeNet_without_BN():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(6, kernel_size=5, padding='same', activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(16, kernel_size=5, activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])

# 使用批量归一化的LeNet
def LeNet_with_BN():
    return tf.keras.models.Sequential([
        # 卷积 → 批量归一化 → 激活
        tf.keras.layers.Conv2D(6, kernel_size=5, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),  # 使用ReLU而非Sigmoid
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),

        tf.keras.layers.Conv2D(16, kernel_size=5),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(120),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(84),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),

        tf.keras.layers.Dense(10)
    ])

# 创建两个模型对比
print("\n创建模型...")
net_without_bn = LeNet_without_BN()
net_with_bn = LeNet_with_BN()

# 测试数据流
X = tf.random.uniform((32, 28, 28, 1))  # batch_size=32

print("\n【不使用批量归一化】")
print("数据流:")
X_test = X
for i, layer in enumerate(net_without_bn.layers):
    X_test = layer(X_test)
    print(f"{i+1}. {layer.__class__.__name__:20s}: {X_test.shape}")

print("\n【使用批量归一化】")
print("数据流:")
X_test = X
for i, layer in enumerate(net_with_bn.layers):
    X_test = layer(X_test)
    print(f"{i+1}. {layer.__class__.__name__:20s}: {X_test.shape}")

# 批量归一化的工作原理演示
print("\n" + "=" * 70)
print("批量归一化的工作原理演示")
print("=" * 70)

# 创建一个简单的批量数据
batch_data = tf.random.normal((4, 3))  # 4个样本，3个特征
print("\n原始批量数据:")
print(batch_data.numpy())

# 手动计算批量归一化
mean = tf.reduce_mean(batch_data, axis=0)
variance = tf.reduce_mean(tf.square(batch_data - mean), axis=0)
std = tf.sqrt(variance + 1e-5)

normalized = (batch_data - mean) / std

print("\n批量均值 μ:")
print(mean.numpy())
print("\n批量标准差 σ:")
print(std.numpy())
print("\n归一化后的数据:")
print(normalized.numpy())
print("\n归一化后的均值（应该接近0）:")
print(tf.reduce_mean(normalized, axis=0).numpy())
print("归一化后的方差（应该接近1）:")
print(tf.reduce_mean(tf.square(normalized), axis=0).numpy())

# 使用TensorFlow的BatchNormalization验证
bn_layer = tf.keras.layers.BatchNormalization()
bn_output = bn_layer(batch_data, training=True)
print("\nTensorFlow BatchNormalization输出:")
print(bn_output.numpy())

print("\n" + "=" * 70)
print("批量归一化的关键要点")
print("=" * 70)
print("""
1. 【位置】
   放在线性层之后、激活函数之前
   卷积 → BN → ReLU

2. 【训练vs测试】
   训练时：使用当前批量的均值和方差
   测试时：使用训练时的移动平均值

3. 【可学习参数】
   γ (gamma): 缩放参数，初始化为1
   β (beta): 平移参数，初始化为0

   这两个参数让网络能够"撤销"归一化（如果需要的话）

4. 【效果】
   ✅ 加速训练（可以用更大的学习率）
   ✅ 减少对权重初始化的依赖
   ✅ 起到正则化作用（类似Dropout）
   ✅ 允许使用饱和激活函数（如Sigmoid）

5. 【注意事项】
   ⚠️ 批量大小太小时效果不好（建议>32）
   ⚠️ 增加了一些计算开销
   ⚠️ 训练和测试行为不同（需要正确设置training参数）
""")

# 模型总结对比
print("\n" + "=" * 70)
print("模型参数对比")
print("=" * 70)
# 先build模型再count_params
net_without_bn.build((None, 28, 28, 1))
net_with_bn.build((None, 28, 28, 1))
print(f"\n不使用BN的模型参数: {net_without_bn.count_params():,}")
print(f"使用BN的模型参数: {net_with_bn.count_params():,}")
print(f"增加的参数: {net_with_bn.count_params() - net_without_bn.count_params()}")
print("（批量归一化增加了γ和β参数，但相对整个网络很少）")

print("\n✅ 7.5 批量归一化代码运行完成!")
