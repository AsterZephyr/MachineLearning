#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""7.6. ResNet - 残差网络"""

import tensorflow as tf

print("=" * 70)
print("7.6. ResNet - 残差网络 (Residual Network)")
print("=" * 70)

print("""
ResNet的核心创新：残差连接（Skip Connection）

问题：网络越深，训练越困难
- 梯度消失/爆炸
- 退化问题：更深的网络表现反而更差

解决方案：残差块
- 让网络学习残差 F(x) = H(x) - x
- 输出 H(x) = F(x) + x
- 即使F(x)学不好，至少还有恒等映射x

数学表达：
H(x) = F(x) + x

其中：
- x: 输入
- F(x): 残差（需要学习的部分）
- H(x): 输出
""")

class Residual(tf.keras.Model):
    """
    残差块

    包含两个3×3卷积层，以及一个跳跃连接
    """
    def __init__(self, num_channels, use_1x1conv=False, strides=1):
        super().__init__()
        # 第一个卷积层
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same', strides=strides)
        self.bn1 = tf.keras.layers.BatchNormalization()

        # 第二个卷积层
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        # 可选的1×1卷积（用于调整维度）
        if use_1x1conv:
            self.conv3 = tf.keras.layers.Conv2D(
                num_channels, kernel_size=1, strides=strides)
        else:
            self.conv3 = None

    def call(self, X):
        # 主路径：卷积 → BN → ReLU → 卷积 → BN
        Y = tf.keras.activations.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))

        # 残差连接：调整X的维度（如果需要）
        if self.conv3:
            X = self.conv3(X)

        # 残差相加
        Y += X

        # 最后再激活
        return tf.keras.activations.relu(Y)

def resnet_block(num_channels, num_residuals, first_block=False):
    """
    ResNet块：包含多个残差块

    参数:
        num_channels: 输出通道数
        num_residuals: 残差块数量
        first_block: 是否是第一个块
    """
    blk = tf.keras.Sequential()
    for i in range(num_residuals):
        if i == 0 and not first_block:
            # 第一个残差块可能需要改变通道数和尺寸
            blk.add(Residual(num_channels, use_1x1conv=True, strides=2))
        else:
            blk.add(Residual(num_channels))
    return blk

def ResNet18():
    """
    构建ResNet-18

    架构：
    - 1个7×7卷积 + BN + ReLU + 池化
    - 4个ResNet块（每块2个残差块）
    - 全局平均池化
    - 全连接层
    """
    return tf.keras.Sequential([
        # 初始模块
        tf.keras.layers.Conv2D(64, kernel_size=7, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same'),

        # ResNet块
        resnet_block(64, 2, first_block=True),  # 64通道，2个残差块
        resnet_block(128, 2),                    # 128通道，2个残差块
        resnet_block(256, 2),                    # 256通道，2个残差块
        resnet_block(512, 2),                    # 512通道，2个残差块

        # 全局平均池化
        tf.keras.layers.GlobalAveragePooling2D(),

        # 输出层
        tf.keras.layers.Dense(10)
    ])

# 创建模型
print("\n构建ResNet-18模型...")
net = ResNet18()

# 测试数据流
X = tf.random.uniform((1, 224, 224, 1))
print("\n数据流动过程:")
print(f"输入: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    print(f"层{i+1} ({layer.__class__.__name__:25s}): {X.shape}")

print("\n" + "=" * 70)
print("残差连接的直观理解")
print("=" * 70)
print("""
【没有残差连接的网络】
x → [Conv → BN → ReLU → Conv → BN → ReLU] → H(x)
     ￣￣￣￣￣￣￣ F(x) ￣￣￣￣￣￣￣

网络需要直接学习 H(x)

【有残差连接的网络】
x → [Conv → BN → ReLU → Conv → BN] → F(x) ┐
 ↓                                          ↓
 └──────────────────────────────────────→ (+) → ReLU → H(x)

网络只需要学习 F(x) = H(x) - x（残差）

为什么更容易训练？
1. 恒等映射：如果某层学不到有用特征，F(x)→0即可
2. 梯度流动：反向传播时梯度可以直接通过跳跃连接传递
3. 避免退化：更深的网络不会比浅网络差
""")

# 单个残差块演示
print("\n" + "=" * 70)
print("单个残差块的详细演示")
print("=" * 70)

# 创建一个残差块
res_block = Residual(64)
X_demo = tf.random.uniform((2, 56, 56, 64))

print(f"\n输入形状: {X_demo.shape}")
print("通过残差块...")

# 跟踪计算过程
Y_conv1 = res_block.conv1(X_demo)
print(f"第1个卷积后: {Y_conv1.shape}")

Y_bn1 = res_block.bn1(Y_conv1)
Y_relu1 = tf.keras.activations.relu(Y_bn1)
print(f"BN + ReLU后: {Y_relu1.shape}")

Y_conv2 = res_block.conv2(Y_relu1)
print(f"第2个卷积后: {Y_conv2.shape}")

Y_bn2 = res_block.bn2(Y_conv2)
print(f"BN后: {Y_bn2.shape}")

# 残差连接
Y_final = Y_bn2 + X_demo  # 关键：加上输入
print(f"加上残差连接后: {Y_final.shape}")

Y_output = tf.keras.activations.relu(Y_final)
print(f"最终输出: {Y_output.shape}")

print("\n✅ 7.6 ResNet代码运行完成!")
print("\nResNet的重要性:")
print("- 解决了深度网络训练困难的问题")
print("- 使得训练上百层的网络成为可能")
print("- ResNet-152在ImageNet上达到了超越人类的表现")
print("- 残差连接成为现代深度学习的标准组件")
