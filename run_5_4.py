#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.4. 自定义层"""

import tensorflow as tf
import numpy as np

print("=" * 50)
print("5.4. 自定义层")
print("=" * 50)

# 5.4.1. 不带参数的层
print("\n5.4.1. 不带参数的层")

class CenteredLayer(tf.keras.Model):
    def __init__(self):
        super().__init__()

    def call(self, inputs):
        return inputs - tf.reduce_mean(inputs)

# 测试层
layer = CenteredLayer()
result = layer(tf.constant([1.0, 2, 3, 4, 5]))
print("输入: [1, 2, 3, 4, 5]")
print("中心化后:", result.numpy())

# 将层作为组件合并到更复杂的模型中
net = tf.keras.Sequential([tf.keras.layers.Dense(128), CenteredLayer()])

# 测试网络
Y = net(tf.random.uniform((4, 8)))
mean_value = tf.reduce_mean(Y)
print(f"\n网络输出的均值: {mean_value.numpy():.10f} (应该接近0)")

# 5.4.2. 带参数的层
print("\n5.4.2. 带参数的层")

class MyDense(tf.keras.Model):
    def __init__(self, units):
        super().__init__()
        self.units = units

    def build(self, X_shape):
        self.weight = self.add_weight(name='weight',
            shape=[X_shape[-1], self.units],
            initializer=tf.random_normal_initializer())
        self.bias = self.add_weight(
            name='bias', shape=[self.units],
            initializer=tf.zeros_initializer())

    def call(self, X):
        linear = tf.matmul(X, self.weight) + self.bias
        return tf.nn.relu(linear)

# 实例化MyDense类并访问其模型参数
dense = MyDense(3)
X = tf.random.uniform((2, 5))
output = dense(X)
print("\n自定义Dense层输出形状:", output.shape)
print("权重形状:", dense.get_weights()[0].shape)
print("偏置形状:", dense.get_weights()[1].shape)

# 使用自定义层直接执行前向传播计算
output2 = dense(tf.random.uniform((2, 5)))
print("\n再次前向传播输出形状:", output2.shape)

# 使用自定义层构建模型
print("\n使用自定义层构建模型:")
net = tf.keras.models.Sequential([MyDense(8), MyDense(1)])
result = net(tf.random.uniform((2, 64)))
print("网络输出形状:", result.shape)

print("\n✅ 5.4 自定义层代码运行完成!")
