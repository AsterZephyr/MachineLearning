#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.6. GPU"""

import tensorflow as tf
from tensorflow.keras import layers

print("=" * 50)
print("5.6. GPU")
print("=" * 50)

print("\nTensorFlow版本:", tf.__version__)

# 5.6.1. 计算设备
print("\n5.6.1. 计算设备")

# 查询可用gpu数量
num_gpus = len(tf.config.list_physical_devices('GPU'))
print(f"可用GPU数量: {num_gpus}")

# 查询所有可用设备
print("\n所有可用设备:")
for device in tf.config.list_physical_devices():
    print(f"  {device}")

# 定义两个便利的函数
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if len(tf.config.list_physical_devices('GPU')) >= i + 1:
        return tf.device(f'/GPU:{i}')
    return tf.device('/CPU:0')

def try_all_gpus():
    """返回所有可用的GPU，如果没有GPU，则返回[cpu()]"""
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    devices = [tf.device(f'/GPU:{i}') for i in range(num_gpus)]
    return devices if devices else [tf.device('/CPU:0')]

print(f"\ntry_gpu()返回: {try_gpu()}")
print(f"try_gpu(10)返回: {try_gpu(10)}")
print(f"try_all_gpus()返回: {try_all_gpus()}")

# 5.6.2. 张量与GPU
print("\n5.6.2. 张量与GPU")

# 默认情况下，张量是在CPU上创建的
x = tf.constant([1, 2, 3])
print(f"\n默认创建的张量设备: {x.device}")

# 5.6.2.1. 存储在GPU上
print("\n5.6.2.1. 存储在GPU上")
with try_gpu():
    X = tf.ones((2, 3))
print("在GPU/CPU上创建的张量:", X.device)

# 5.6.2.2. 复制
print("\n5.6.2.2. 复制")
if num_gpus >= 2:
    with try_gpu(1):
        Y = tf.random.uniform((2, 3))
    print("在第二个GPU上创建的张量:", Y.device)
else:
    print("系统只有少于2个GPU，跳过多GPU测试")

# 在同一设备上的张量
with try_gpu():
    Z = X
print("\n复制到同一设备:", Z.device)

# 5.6.2.3. 旁注
print("\n5.6.2.3. 旁注")
with try_gpu():
    Z2 = Z
print(f"张量已在目标设备上，Z2 is Z: {Z2 is Z}")

# 5.6.3. 神经网络与GPU
print("\n5.6.3. 神经网络与GPU")

# 创建简单的网络
if num_gpus > 0:
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(1)
        ])
    print("使用MirroredStrategy创建网络")
else:
    net = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1)
    ])
    print("在CPU上创建网络")

# 当输入为GPU上的张量时，模型将在同一GPU上计算结果
with try_gpu():
    result = net(X)
print(f"\n网络输出设备: {result.device}")

# 确认模型参数存储在同一个设备上
if len(net.layers[0].weights) > 0:
    # 获取权重张量并访问其device属性
    weight_tensor = tf.convert_to_tensor(net.layers[0].weights[0])
    print(f"模型参数设备: {weight_tensor.device}")
else:
    # 需要先调用网络以初始化参数
    net(X)
    weight_tensor = tf.convert_to_tensor(net.layers[0].weights[0])
    print(f"模型参数设备: {weight_tensor.device}")

print("\n✅ 5.6 GPU代码运行完成!")
if num_gpus == 0:
    print("注意: 本机没有可用的GPU，所有计算都在CPU上进行")
