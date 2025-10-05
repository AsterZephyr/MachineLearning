#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.1. 从全连接层到卷积"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("6.1. 从全连接层到卷积")
print("=" * 50)

# 6.1.1. 全连接层的问题
print("\n6.1.1. 全连接层的问题")
input_size = 1000 * 1000
hidden_size = 1000
num_params = input_size * hidden_size
print(f"全连接层参数数量: {num_params:,}")
print(f"这相当于 {num_params / 1e9:.1f} 十亿个参数！")

# 6.1.2. 卷积层的优势
print("\n6.1.2. 卷积层的优势")
kernel_size = 5
num_filters = 64
conv_params = kernel_size * kernel_size * num_filters
print(f"卷积层参数数量: {conv_params:,}")
print(f"参数减少了 {num_params / conv_params:,.0f} 倍！")

# 6.1.3. 平移不变性
print("\n6.1.3. 平移不变性演示")
def create_image_with_point(position):
    img = np.zeros((8, 8))
    img[position[0], position[1]] = 1.0
    return img

img1 = create_image_with_point((3, 3))
img2 = create_image_with_point((3, 5))

print("图像1 (亮点在中间):")
print(img1.astype(int))
print("\n图像2 (亮点在右边):")
print(img2.astype(int))

# 6.1.4. 局部性
print("\n6.1.4. 局部性演示")
def show_receptive_field(img_size=8, kernel_size=3):
    img = np.zeros((img_size, img_size))
    start = img_size // 2 - kernel_size // 2
    end = start + kernel_size
    img[start:end, start:end] = 1.0
    return img

receptive_field = show_receptive_field()
print("3x3 卷积核的局部感受野（1表示关注的区域）:")
print(receptive_field.astype(int))

print("\n✅ 6.1 从全连接层到卷积代码运行完成!")
