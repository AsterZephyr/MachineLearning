#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.3. 延后初始化"""

import tensorflow as tf

print("=" * 50)
print("5.3. 延后初始化")
print("=" * 50)

# 创建网络，不指定输入维度
net = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation=tf.nn.relu),
    tf.keras.layers.Dense(10),
])

print("\n网络已创建，但参数尚未初始化")

# 此时访问参数会发现权重还未初始化
print("\n未初始化的参数:")
try:
    weights = [net.layers[i].get_weights() for i in range(len(net.layers))]
    print("参数:", weights)
except:
    print("参数尚未初始化，因为网络还没有处理过数据")

# 实例化网络
print("\n实例化网络 - 通过传入数据来初始化")
X = tf.random.uniform((2, 20))
Y = net(X)
print("输出形状:", Y.shape)

# 现在参数已经初始化
print("\n参数已初始化:")
weights = [w.shape for w in net.get_weights()]
print("参数形状:", weights)

print("\n第一层权重形状:", net.layers[0].weights[0].shape)
print("第一层偏置形状:", net.layers[0].weights[1].shape)
print("第二层权重形状:", net.layers[1].weights[0].shape)
print("第二层偏置形状:", net.layers[1].weights[1].shape)

print("\n✅ 5.3 延后初始化代码运行完成!")
