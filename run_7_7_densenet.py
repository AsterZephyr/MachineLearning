#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.7. DenseNet - 稠密连接网络"""

import tensorflow as tf

print("=" * 70)
print("7.7. DenseNet - 稠密连接网络 (Densely Connected Network)")
print("=" * 70)

print("""
DenseNet的核心创新：稠密连接（Dense Connection）

ResNet vs DenseNet:
- ResNet: H(x) = F(x) + x         (加法，残差连接)
- DenseNet: H(x) = [x, F1(x), F2(x), ...] (拼接，稠密连接)

关键区别：
- ResNet：通过相加组合特征
- DenseNet：通过拼接保留所有特征

优势：
1. 缓解梯度消失问题
2. 加强特征传播
3. 鼓励特征复用
4. 大幅减少参数数量
""")

def conv_block(num_channels):
    """
    卷积块：BN → ReLU → 3×3卷积

    这是DenseNet的基本构建单元
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Conv2D(num_channels, kernel_size=3, padding='same')
    ])

class DenseBlock(tf.keras.layers.Layer):
    """
    稠密块：包含多个卷积块，每个卷积块都与前面所有块连接

    特点：
    - 每层的输入是前面所有层的输出拼接
    - 通道数随深度增加：输入 → 输入+k → 输入+2k → ...
    """
    def __init__(self, num_convs, num_channels):
        super().__init__()
        self.convs = []
        for _ in range(num_convs):
            self.convs.append(conv_block(num_channels))

    def call(self, X):
        # 逐层添加新特征，并与之前的特征拼接
        for conv in self.convs:
            Y = conv(X)
            # 在通道维度拼接输入和输出
            X = tf.concat([X, Y], axis=-1)
        return X

def transition_block(num_channels):
    """
    过渡层：连接两个稠密块

    作用：
    1. 1×1卷积降维（减少通道数）
    2. 2×2平均池化（减小尺寸）
    """
    return tf.keras.models.Sequential([
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        # 1×1卷积降维
        tf.keras.layers.Conv2D(num_channels, kernel_size=1),
        # 平均池化减小尺寸
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2)
    ])

def DenseNet():
    """
    构建DenseNet

    架构：
    - 初始卷积层
    - 4个稠密块（每块4个卷积层，增长率32）
    - 3个过渡层（连接前3个稠密块）
    - 全局平均池化
    - 全连接层
    """
    # 初始卷积
    net = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same')
    ])

    # 稠密块配置：(卷积层数, 增长率)
    num_channels = 64  # 初始通道数
    growth_rate = 32   # 增长率：每层添加的通道数
    num_convs_in_dense_blocks = [4, 4, 4, 4]

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        # 添加稠密块
        net.add(DenseBlock(num_convs, growth_rate))
        # 更新通道数：原有通道 + 新增通道
        num_channels += num_convs * growth_rate

        # 最后一个稠密块后不添加过渡层
        if i != len(num_convs_in_dense_blocks) - 1:
            # 过渡层将通道数减半
            num_channels //= 2
            net.add(transition_block(num_channels))

    # 全局平均池化 + 全连接层
    net.add(tf.keras.layers.BatchNormalization())
    net.add(tf.keras.layers.Activation('relu'))
    net.add(tf.keras.layers.GlobalAveragePooling2D())
    net.add(tf.keras.layers.Dense(10))

    return net

# 创建模型
print("\n构建DenseNet模型...")
net = DenseNet()

# 测试数据流
X = tf.random.uniform((1, 96, 96, 1))
print("\n数据流动过程:")
print(f"输入: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    layer_type = layer.__class__.__name__
    if 'DenseBlock' in layer_type:
        print(f"稠密块{i+1}: {X.shape} (通道数增加)")
    elif 'Sequential' in layer_type:
        if i < 5:
            print(f"初始模块: {X.shape}")
        else:
            print(f"过渡层{i+1}: {X.shape} (降维+池化)")
    else:
        print(f"层{i+1} ({layer_type:25s}): {X.shape}")

print("\n" + "=" * 70)
print("DenseNet的稠密连接示意")
print("=" * 70)
print("""
【传统卷积网络】
x0 → [Conv1] → x1 → [Conv2] → x2 → [Conv3] → x3

【ResNet（残差连接）】
x0 → [Conv1] → x1 ─┐
                    ├→ (+) → [Conv2] → x2
           x0 ──────┘

【DenseNet（稠密连接）】
x0 → [Conv1] → x1 ─┬─────────────┬─→ [x0,x1,x2,x3]
                    ↓             ↓
           x0 ─→ [Conv2] → x2 ────┤
                              ↓    ↓
                  x0,x1 ─→ [Conv3] → x3

每一层都接收前面所有层的输出（通过拼接）
""")

print("\n" + "=" * 70)
print("稠密块详细演示")
print("=" * 70)

# 创建一个小的稠密块演示
dense_block_demo = DenseBlock(num_convs=3, num_channels=4)
X_demo = tf.random.uniform((1, 8, 8, 8))  # 初始8个通道

print(f"\n输入形状: {X_demo.shape} (8个通道)")
print("通过3层稠密块，每层增长4个通道...\n")

X_temp = X_demo
for i, conv in enumerate(dense_block_demo.convs):
    Y = conv(X_temp)
    print(f"第{i+1}层卷积输出: {Y.shape} (新增4个通道)")
    X_temp = tf.concat([X_temp, Y], axis=-1)
    print(f"  拼接后: {X_temp.shape} (总通道数 = {X_temp.shape[-1]})")

print("\n通道数变化: 8 → 12 → 16 → 20")
print("公式: 输出通道 = 输入通道 + (卷积层数 × 增长率)")
print("      20 = 8 + (3 × 4) ✓")

print("\n" + "=" * 70)
print("DenseNet的关键优势")
print("=" * 70)
print("""
1. 【参数效率】
   - 增长率通常很小（12-32）
   - 每层只产生少量特征图
   - 通过特征复用大幅减少参数

2. 【特征复用】
   - 所有层共享特征
   - 避免重复学习相同特征
   - 提高特征利用效率

3. 【梯度流动】
   - 每层都有直接梯度路径到损失函数
   - 更容易训练深层网络
   - 缓解梯度消失问题

4. 【隐式深度监督】
   - 所有层都能直接访问损失函数
   - 类似于多个中间分类器
   - 提高训练效果

5. 【正则化效果】
   - 参数共享起到正则化作用
   - 减少过拟合风险
   - 小数据集上也能工作良好
""")

print("\n" + "=" * 70)
print("ResNet vs DenseNet对比")
print("=" * 70)
print("""
特性              ResNet              DenseNet
─────────────────────────────────────────────────
连接方式          相加 (+)            拼接 (concat)
特征组合          融合特征            保留所有特征
通道数变化        不变                线性增长
参数量            中等                较少
内存占用          较少                较多（存储特征图）
梯度流动          skip connection     dense connection
计算效率          高                  中等
特征复用          隐式                显式

适用场景：
- ResNet：需要高效推理的场景
- DenseNet：需要参数效率和特征复用的场景
""")

# 模型总结
print("\n" + "=" * 70)
print("DenseNet模型架构总结")
print("=" * 70)
net.summary()

print("\n✅ 7.7 DenseNet代码运行完成!")
print("\nDenseNet的影响：")
print("- 证明了特征复用的重要性")
print("- 启发了后续的高效网络设计（如MobileNet、EfficientNet）")
print("- 在参数效率和性能之间取得了良好平衡")
print("- 是现代深度学习架构设计的重要里程碑")
