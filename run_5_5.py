#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""5.5. 读写文件"""

import numpy as np
import tensorflow as tf
import os

print("=" * 50)
print("5.5. 读写文件")
print("=" * 50)

# 5.5.1. 加载和保存张量
print("\n5.5.1. 加载和保存张量")

# 保存单个张量
x = tf.range(4)
np.save('x-file.npy', x)
print("保存张量 x:", x.numpy())

# 从文件中读回数据
x2 = np.load('x-file.npy', allow_pickle=True)
print("读取张量 x2:", x2)

# 存储一个张量列表
y = tf.zeros(4)
np.save('xy-files.npy', [x, y])
print("\n保存张量列表 [x, y]")

# 读取列表
x2, y2 = np.load('xy-files.npy', allow_pickle=True)
print("读取张量 x2:", x2)
print("读取张量 y2:", y2)

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
np.save('mydict.npy', mydict)
print("\n保存字典:", mydict)

# 读取字典
mydict2 = np.load('mydict.npy', allow_pickle=True)
print("读取字典:", mydict2.item())

# 5.5.2. 加载和保存模型参数
print("\n5.5.2. 加载和保存模型参数")

class MLP(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        return self.out(x)

net = MLP()
X = tf.random.uniform((2, 20))
Y = net(X)
print("原始网络输出形状:", Y.shape)

# 将模型的参数存储为一个叫做"mlp.params.weights.h5"的文件
net.save_weights('mlp.params.weights.h5')
print("\n模型参数已保存到 mlp.params.weights.h5")

# 实例化原始多层感知机模型的一个克隆
clone = MLP()
clone(X)  # 需要先调用以初始化参数
clone.load_weights('mlp.params.weights.h5')
print("模型参数已加载到克隆网络")

# 验证两个模型的参数是否相同
Y_clone = clone(X)
print("\n验证克隆网络输出形状:", Y_clone.shape)
print("原始输出和克隆输出是否相等:", tf.reduce_all(Y == Y_clone).numpy())

# 清理临时文件
for f in ['x-file.npy', 'xy-files.npy', 'mydict.npy']:
    if os.path.exists(f):
        os.remove(f)
        print(f"\n清理文件: {f}")

print("\n✅ 5.5 读写文件代码运行完成!")
