#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.4. GoogLeNet - 含并行连结的网络"""

import tensorflow as tf

print("=" * 70)
print("7.4. GoogLeNet - 含并行连结的网络")
print("=" * 70)

print("""
GoogLeNet（也叫Inception v1）的核心创新：
1. Inception块：并行使用不同大小的卷积核
2. 使用1×1卷积降维，减少计算量
3. 在输出维度上连结不同路径
4. 全局平均池化代替全连接层

Inception块的思想：
"我不知道用3×3还是5×5卷积核，那就都用，然后连结起来！"
""")

class Inception(tf.keras.Model):
    """
    Inception块：4条并行路径

    路径1: 1×1卷积
    路径2: 1×1卷积 → 3×3卷积
    路径3: 1×1卷积 → 5×5卷积
    路径4: 3×3最大池化 → 1×1卷积
    """
    def __init__(self, c1, c2, c3, c4):
        super().__init__()
        # 路径1: 1×1卷积
        self.p1_1 = tf.keras.layers.Conv2D(c1, 1, activation='relu')

        # 路径2: 1×1卷积 → 3×3卷积
        self.p2_1 = tf.keras.layers.Conv2D(c2[0], 1, activation='relu')
        self.p2_2 = tf.keras.layers.Conv2D(
            c2[1], 3, padding='same', activation='relu')

        # 路径3: 1×1卷积 → 5×5卷积
        self.p3_1 = tf.keras.layers.Conv2D(c3[0], 1, activation='relu')
        self.p3_2 = tf.keras.layers.Conv2D(
            c3[1], 5, padding='same', activation='relu')

        # 路径4: 3×3最大池化 → 1×1卷积
        self.p4_1 = tf.keras.layers.MaxPool2D(3, 1, padding='same')
        self.p4_2 = tf.keras.layers.Conv2D(c4, 1, activation='relu')

    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        # 在通道维度连结
        return tf.concat([p1, p2, p3, p4], axis=-1)

# 构建GoogLeNet
def GoogLeNet():
    # 模块1
    b1 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

    # 模块2
    b2 = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, 1, activation='relu'),
        tf.keras.layers.Conv2D(192, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

    # 模块3: 2个Inception块
    b3 = tf.keras.models.Sequential([
        Inception(64, (96, 128), (16, 32), 32),
        Inception(128, (128, 192), (32, 96), 64),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

    # 模块4: 5个Inception块
    b4 = tf.keras.models.Sequential([
        Inception(192, (96, 208), (16, 48), 64),
        Inception(160, (112, 224), (24, 64), 64),
        Inception(128, (128, 256), (24, 64), 64),
        Inception(112, (144, 288), (32, 64), 64),
        Inception(256, (160, 320), (32, 128), 128),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

    # 模块5: 2个Inception块 + 全局平均池化
    b5 = tf.keras.models.Sequential([
        Inception(256, (160, 320), (32, 128), 128),
        Inception(384, (192, 384), (48, 128), 128),
        tf.keras.layers.GlobalAveragePooling2D()
    ])

    # 组合所有模块
    model = tf.keras.models.Sequential([b1, b2, b3, b4, b5,
                                        tf.keras.layers.Dense(10)])
    return model

# 创建模型
print("\n构建GoogLeNet模型...")
net = GoogLeNet()

# 查看数据流
X = tf.random.uniform((1, 96, 96, 1))
print("\n数据流动过程:")
print(f"输入: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    print(f"模块{i+1}: {X.shape}")

print("\n" + "=" * 70)
print("Inception块的设计思想")
print("=" * 70)
print("""
【问题】如何选择卷积核大小？
- 1×1 卷积：捕获点信息
- 3×3 卷积：捕获局部信息
- 5×5 卷积：捕获更大范围信息
- 池化层：降低空间分辨率

【Inception的解决方案】全部都要！

输入 (28×28×256)
    ↓
┌────┴────┬────────┬────────┬────────┐
│ 1×1卷积  │1×1→3×3 │1×1→5×5 │池化→1×1│
│ (64通道) │(128通道)│(32通道)│(32通道)│
└────┬────┴────────┴────────┴────────┘
     ↓
连结 (28×28×256)  # 64+128+32+32=256

关键：使用1×1卷积降维
- 路径2: 先用1×1卷积从256→96通道，再3×3卷积到128
- 这样比直接256→128的3×3卷积参数少很多！
""")

# 模型总结
print("\n" + "=" * 70)
print("GoogLeNet模型架构总结")
print("=" * 70)
net.summary()

print("\n✅ 7.4 GoogLeNet代码运行完成!")
