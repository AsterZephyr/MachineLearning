#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.2. 参数管理"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("5.2. 参数管理")
print("=" * 50)

# 创建一个简单的Sequential网络
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation=tf.nn.relu),
    tf.keras.layers.Dense(1),
])

X = tf.random.uniform((2, 4))
print("\n前向传播:")
print(net(X))

# 5.2.1. 参数访问
print("\n5.2.1. 参数访问")
print("第二层的参数:")
print(net.layers[2].weights)

print("\n访问目标参数:")
print(f"类型: {type(net.layers[2].weights[1])}")
print(f"偏置: {net.layers[2].weights[1]}")

# 5.2.1.1. 一次性访问所有参数
print("\n5.2.1.1. 一次性访问所有参数")
print("第一个全连接层的参数:")
print(net.layers[1].weights[0])

# 5.2.1.2. 从嵌套块收集参数
print("\n5.2.1.2. 从嵌套块收集参数")
def block1():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation=tf.nn.relu)
    ])

def block2():
    net = tf.keras.Sequential()
    for i in range(4):
        net.add(block1())
    return net

rgnet = tf.keras.Sequential()
rgnet.add(block2())
rgnet.add(tf.keras.layers.Dense(10))
print("嵌套网络输出:")
print(rgnet(X))

# 5.2.2. 参数初始化
print("\n5.2.2. 参数初始化")
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.01),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1)
])

net(X)
print("正态分布初始化的权重:")
print(net.layers[1].weights[0])
print("零初始化的偏置:")
print(net.layers[1].weights[1])

# 5.2.2.1. 内置初始化
print("\n5.2.2.1. 内置初始化 - 常数初始化")
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4, activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.Constant(1),
        bias_initializer=tf.zeros_initializer()),
    tf.keras.layers.Dense(1),
])

net(X)
print("常数初始化的权重:")
print(net.layers[1].weights[0])

# Xavier初始化
print("\nXavier初始化:")
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=tf.keras.initializers.GlorotUniform()),
    tf.keras.layers.Dense(
        1, kernel_initializer=tf.keras.initializers.GlorotUniform()),
])

net(X)
print("Xavier初始化的权重:")
print(net.layers[1].weights[0])

# 5.2.2.2. 自定义初始化
print("\n5.2.2.2. 自定义初始化")
class MyInit(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        data = tf.random.uniform(shape, -10, 10, dtype=dtype)
        factor = tf.cast(tf.abs(data) >= 5, dtype=dtype)
        return data * factor

net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        4,
        activation=tf.nn.relu,
        kernel_initializer=MyInit()),
    tf.keras.layers.Dense(1),
])

net(X)
print("自定义初始化的权重:")
print(net.layers[1].weights[0])

# 直接设置参数
net.layers[1].weights[0][:].assign(net.layers[1].weights[0] + 1)
net.layers[1].weights[0][0, 0].assign(42)
print("\n直接设置参数后:")
print(net.layers[1].weights[0])

# 5.2.3. 参数绑定
print("\n5.2.3. 参数绑定")
shared = tf.keras.layers.Dense(4, activation=tf.nn.relu)
net = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    shared,
    shared,
    tf.keras.layers.Dense(1),
])

net(X)
print(f"共享层检查: 层数 = {len(net.layers)}")
print(f"参数是否相同: {net.layers[1].weights[0] is net.layers[2].weights[0]}")

print("\n✅ 5.2 参数管理代码运行完成!")
