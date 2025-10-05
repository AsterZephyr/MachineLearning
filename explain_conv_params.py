#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""深入理解卷积如何减少参数"""

import tensorflow as tf
import numpy as np

print("=" * 70)
print("深入理解：卷积如何大幅减少参数")
print("=" * 70)

# ============================================================================
# 第一部分：全连接层 vs 卷积层的参数对比
# ============================================================================
print("\n【第一部分】参数数量对比")
print("-" * 70)

# 假设一个小图像：8x8像素
img_h, img_w = 8, 8
input_pixels = img_h * img_w  # 64个像素

print(f"\n输入图像大小: {img_h}×{img_w} = {input_pixels}个像素")

# 1. 全连接层的做法
print("\n1️⃣ 【全连接层】每个输出神经元连接到所有输入像素")
print("   假设输出也是8×8个神经元")

output_neurons = 64  # 8x8 = 64个输出神经元
fc_params = input_pixels * output_neurons
print(f"   参数数量 = {input_pixels} × {output_neurons} = {fc_params:,}个参数")

# 2. 卷积层的做法
print("\n2️⃣ 【卷积层】使用3×3的卷积核")
kernel_h, kernel_w = 3, 3
kernel_params = kernel_h * kernel_w
num_filters = 1  # 先考虑1个滤波器

print(f"   卷积核大小: {kernel_h}×{kernel_w} = {kernel_params}个参数")
print(f"   这个卷积核在整个图像上滑动，共享这{kernel_params}个参数")
print(f"   参数数量 = {kernel_params}个参数")

print(f"\n   💡 参数减少: {fc_params} → {kernel_params}")
print(f"   减少了 {fc_params / kernel_params:.0f} 倍！")

# ============================================================================
# 第二部分：可视化参数共享机制
# ============================================================================
print("\n" + "=" * 70)
print("【第二部分】参数共享的具体机制")
print("-" * 70)

# 创建一个简单的5×5图像
X = np.arange(25).reshape(5, 5).astype(float)
print("\n输入图像 (5×5):")
print(X)

# 创建一个3×3卷积核
K = np.array([[1, 0, -1],
              [2, 0, -2],
              [1, 0, -1]], dtype=float)
print("\n卷积核 (3×3) - Sobel边缘检测器:")
print(K)
print("\n这个卷积核只有9个参数，但是会在整个图像上滑动!")

# 手动演示卷积过程
print("\n" + "-" * 70)
print("【卷积核滑动过程演示】")
print("-" * 70)

def show_conv_step(X, K, i, j):
    """展示某个位置的卷积计算"""
    h, w = K.shape
    window = X[i:i+h, j:j+w]
    result = np.sum(window * K)

    print(f"\n位置 ({i},{j}):")
    print(f"  输入窗口:        卷积核:         逐元素相乘:")
    for row in range(h):
        window_row = "  " + " ".join([f"{window[row,col]:5.0f}" for col in range(w)])
        kernel_row = "  " + " ".join([f"{K[row,col]:5.0f}" for col in range(w)])
        product_row = "  " + " ".join([f"{window[row,col]*K[row,col]:5.0f}" for col in range(w)])
        print(f"{window_row}    {kernel_row}    {product_row}")
    print(f"  求和: {result:.0f}")
    return result

# 演示4个位置
output = []
positions = [(0, 0), (0, 1), (1, 0), (1, 1)]

for i, j in positions:
    result = show_conv_step(X, K, i, j)
    output.append(result)

print("\n" + "-" * 70)
print("💡 关键点：同样的9个参数（卷积核）被重复使用了4次！")
print("   这就是参数共享！")

# ============================================================================
# 第三部分：完整卷积输出
# ============================================================================
print("\n" + "=" * 70)
print("【第三部分】完整的卷积输出")
print("-" * 70)

def manual_conv2d(X, K):
    """手动实现卷积"""
    h, w = K.shape
    out_h = X.shape[0] - h + 1
    out_w = X.shape[1] - w + 1
    Y = np.zeros((out_h, out_w))

    for i in range(out_h):
        for j in range(out_w):
            Y[i, j] = np.sum(X[i:i+h, j:j+w] * K)
    return Y

Y = manual_conv2d(X, K)
print(f"\n输入图像 ({X.shape[0]}×{X.shape[1]}):")
print(X)
print(f"\n卷积核 ({K.shape[0]}×{K.shape[1]}):")
print(K)
print(f"\n输出特征图 ({Y.shape[0]}×{Y.shape[1]}):")
print(Y)

print("\n💡 注意：")
print(f"   - 卷积核在输入上滑动了 {Y.shape[0]}×{Y.shape[1]} = {Y.size} 次")
print(f"   - 但参数始终是那 {K.size} 个！")
print(f"   - 如果用全连接层，需要 {X.size * Y.size} 个参数")

# ============================================================================
# 第四部分：多通道的情况
# ============================================================================
print("\n" + "=" * 70)
print("【第四部分】多通道情况下的参数计算")
print("-" * 70)

print("\n假设：RGB图像 32×32×3 → 输出 64个通道")
print()

# 全连接层
input_size = 32 * 32 * 3
output_size = 32 * 32 * 64
fc_params_multi = input_size * output_size
print(f"1️⃣ 全连接层参数:")
print(f"   {input_size:,} × {output_size:,} = {fc_params_multi:,} 个参数")

# 卷积层
kernel_h, kernel_w = 3, 3
in_channels = 3
out_channels = 64
conv_params_multi = kernel_h * kernel_w * in_channels * out_channels + out_channels
print(f"\n2️⃣ 卷积层参数 (3×3卷积核):")
print(f"   卷积核: {kernel_h}×{kernel_w}×{in_channels}×{out_channels} = {kernel_h * kernel_w * in_channels * out_channels:,}")
print(f"   偏置:   {out_channels}")
print(f"   总计:   {conv_params_multi:,} 个参数")

print(f"\n   💡 参数减少: {fc_params_multi:,} → {conv_params_multi:,}")
print(f"   减少了 {fc_params_multi / conv_params_multi:,.0f} 倍！")

# ============================================================================
# 第五部分：用TensorFlow验证
# ============================================================================
print("\n" + "=" * 70)
print("【第五部分】TensorFlow验证")
print("-" * 70)

# 全连接层
fc_layer = tf.keras.layers.Dense(output_size, input_shape=(input_size,))
fc_layer.build((None, input_size))
print(f"\n全连接层参数数量: {fc_layer.count_params():,}")

# 卷积层
conv_layer = tf.keras.layers.Conv2D(
    filters=64,
    kernel_size=3,
    input_shape=(32, 32, 3)
)
conv_layer.build((None, 32, 32, 3))
print(f"卷积层参数数量:   {conv_layer.count_params():,}")

print(f"\n减少比例: {fc_layer.count_params() / conv_layer.count_params():,.0f}x")

# ============================================================================
# 第六部分：参数共享的直观理解
# ============================================================================
print("\n" + "=" * 70)
print("【第六部分】参数共享的直观理解")
print("-" * 70)

print("\n🔍 全连接层：")
print("   每个输出位置都有自己独特的一套权重")
print("   就像每个员工都有自己的工具箱")
print()
print("   输出[0,0] 用权重集合A (64个参数)")
print("   输出[0,1] 用权重集合B (64个参数)")
print("   输出[0,2] 用权重集合C (64个参数)")
print("   ... 总共需要 64×1024 = 65,536 套权重")

print("\n🔍 卷积层：")
print("   所有输出位置共享同一个卷积核")
print("   就像所有员工共用一个工具箱")
print()
print("   输出[0,0] 用卷积核K (9个参数)")
print("   输出[0,1] 用卷积核K (9个参数) ← 同一个K！")
print("   输出[0,2] 用卷积核K (9个参数) ← 同一个K！")
print("   ... 始终只需要 9 个参数")

print("\n" + "=" * 70)
print("总结：卷积为什么能减少参数？")
print("=" * 70)
print("""
1️⃣ 局部连接：
   - 每个输出只看输入的一小块区域（感受野）
   - 不需要连接所有输入像素

2️⃣ 参数共享：
   - 同一个卷积核在整个图像上滑动
   - 所有位置共享这一套参数

3️⃣ 平移不变性：
   - 同样的特征（如边缘）无论出现在哪里
   - 都用同一个卷积核检测

这就像：
- 全连接 = 每个位置都配一个专属侦探（参数多）
- 卷积    = 一个侦探巡逻整个区域（参数少）
""")

print("\n✅ 理解卷积参数减少的核心：")
print("   卷积核 = 可重复使用的特征检测器")
print("   参数共享 = 核心机制")
print("   局部连接 = 设计原则")
