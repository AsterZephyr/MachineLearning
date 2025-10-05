#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.3. NiN - 网络中的网络"""

import tensorflow as tf

print("=" * 70)
print("7.3. NiN - 网络中的网络 (Network in Network)")
print("=" * 70)

print("""
NiN的核心创新：
1. 使用1×1卷积层代替全连接层
2. 每个NiN块 = 普通卷积 + 两个1×1卷积
3. 使用全局平均池化代替全连接层
4. 大幅减少参数数量

1×1卷积的作用：
- 在每个像素位置添加全连接层
- 增加非线性
- 跨通道信息交互
- 降维或升维
""")

def nin_block(num_channels, kernel_size, strides, padding):
    """
    NiN块 = 一个普通卷积 + 两个1×1卷积

    参数:
        num_channels: 输出通道数
        kernel_size: 卷积核大小
        strides: 步幅
        padding: 填充
    """
    return tf.keras.models.Sequential([
        # 普通卷积层
        tf.keras.layers.Conv2D(
            num_channels, kernel_size=kernel_size,
            strides=strides, padding=padding, activation='relu'),
        # 1×1卷积（相当于逐像素的全连接层）
        tf.keras.layers.Conv2D(
            num_channels, kernel_size=1, activation='relu'),
        # 再一个1×1卷积
        tf.keras.layers.Conv2D(
            num_channels, kernel_size=1, activation='relu'),
    ])

def NiN():
    """构建NiN网络"""
    return tf.keras.models.Sequential([
        # NiN块1
        nin_block(96, kernel_size=11, strides=4, padding='valid'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # NiN块2
        nin_block(256, kernel_size=5, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # NiN块3
        nin_block(384, kernel_size=3, strides=1, padding='same'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2),

        # Dropout
        tf.keras.layers.Dropout(0.5),

        # NiN块4：输出通道数=类别数
        nin_block(10, kernel_size=3, strides=1, padding='same'),

        # 全局平均池化：将每个通道的所有值求平均
        tf.keras.layers.GlobalAveragePooling2D(),

        # 输出已经是(batch_size, 10)，不需要额外的全连接层！
    ])

# 创建模型
print("\n构建NiN模型...")
net = NiN()

# 查看数据流
X = tf.random.uniform((1, 224, 224, 1))
print("\n数据流动过程:")
print(f"输入: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    print(f"第{i+1}层 ({layer.__class__.__name__:20s}): {X.shape}")

print("\n" + "=" * 70)
print("NiN vs 传统CNN的对比")
print("=" * 70)

print("""
【传统CNN（如AlexNet）】
卷积层提取特征 → 展平 → 全连接层 → 输出

问题：
- 全连接层参数过多
- 容易过拟合
- 计算量大

【NiN的改进】
卷积层提取特征 → 1×1卷积（替代全连接）→ 全局平均池化 → 输出

优势：
- 没有全连接层，参数大幅减少
- 1×1卷积保留空间结构
- 全局平均池化对位置更鲁棒
- 减少过拟合风险
""")

# 模型总结
print("\n" + "=" * 70)
print("NiN模型架构总结")
print("=" * 70)
net.summary()

print("\n" + "=" * 70)
print("1×1卷积的直观理解")
print("=" * 70)
print("""
假设输入特征图: 28×28×256（高×宽×通道）

【1×1卷积的计算】
- 卷积核: 1×1×256×128（输出128通道）
- 在每个像素位置（28×28个位置）：
  1. 取出256个通道的值
  2. 与256×128的权重矩阵相乘
  3. 得到128个输出值

等价于：
- 在每个像素位置应用一个256→128的全连接层
- 所有像素位置共享这个全连接层的权重
- 这就是"网络中的网络"的含义！

参数量对比：
- 全连接层: 28×28×256 × 128 = 25,690,112 个参数
- 1×1卷积: 256 × 128 = 32,768 个参数（共享权重）
- 减少了约 784 倍！
""")

print("\n✅ 7.3 NiN代码运行完成!")
