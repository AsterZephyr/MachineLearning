#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""6.6. LeNet"""

import tensorflow as tf
import numpy as np
import time

print("=" * 50)
print("6.6. LeNet - 卷积神经网络")
print("=" * 50)

# 6.6.1. LeNet架构
print("\n6.6.1. 构建LeNet模型")

def LeNet():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation='sigmoid',
                               padding='same'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation='sigmoid'),
        tf.keras.layers.AvgPool2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(120, activation='sigmoid'),
        tf.keras.layers.Dense(84, activation='sigmoid'),
        tf.keras.layers.Dense(10)
    ])

net = LeNet()
X = tf.random.uniform((1, 28, 28, 1))
Y = net(X)

print("LeNet模型架构:")
net.summary()

# 6.6.2. 数据流动可视化
print("\n6.6.2. 数据流动可视化")
X = tf.random.uniform((1, 28, 28, 1))
print(f"输入形状: {X.shape}")

for i, layer in enumerate(net.layers):
    X = layer(X)
    print(f'{layer.__class__.__name__} 输出形状: {X.shape}')

# 6.6.3. 在Fashion-MNIST上训练
print("\n6.6.3. 在Fashion-MNIST上训练（简化版，仅3个epoch）")

# 加载数据
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = np.expand_dims(x_train.astype(np.float32) / 255.0, axis=-1)
x_test = np.expand_dims(x_test.astype(np.float32) / 255.0, axis=-1)

print(f"训练集形状: {x_train.shape}")
print(f"测试集形状: {x_test.shape}")

# 创建数据集
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(10000).batch(batch_size)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(batch_size)

# 训练函数
def train_epoch(net, train_iter, loss, optimizer):
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0

    for X, y in train_iter:
        with tf.GradientTape() as tape:
            y_hat = net(X, training=True)
            l = loss(y, y_hat)

        grads = tape.gradient(l, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))

        total_loss += l
        total_acc += tf.reduce_sum(
            tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, tf.int64), dtype=tf.float32))
        num_batches += 1

    return total_loss / num_batches, total_acc / len(x_train)

# 评估函数
def evaluate_accuracy(net, data_iter):
    total_acc = 0.0
    num_examples = 0

    for X, y in data_iter:
        y_hat = net(X, training=False)
        total_acc += tf.reduce_sum(
            tf.cast(tf.argmax(y_hat, axis=1) == tf.cast(y, tf.int64), dtype=tf.float32))
        num_examples += X.shape[0]

    return total_acc / num_examples

# 训练模型
net = LeNet()
lr = 0.9
num_epochs = 3  # 简化版本，只训练3个epoch
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.SGD(learning_rate=lr)

print("\n开始训练...")
for epoch in range(num_epochs):
    start = time.time()
    train_loss, train_acc = train_epoch(net, train_dataset, loss, optimizer)
    test_acc = evaluate_accuracy(net, test_dataset)

    print(f'epoch {epoch + 1}, '
          f'loss {float(train_loss):.3f}, '
          f'train acc {float(train_acc):.3f}, '
          f'test acc {float(test_acc):.3f}, '
          f'time {time.time() - start:.1f}s')

# 6.6.4. 模型预测
print("\n6.6.4. 模型预测示例")

text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']

X_sample = x_test[:10]
y_sample = y_test[:10]
y_pred = tf.argmax(net(X_sample, training=False), axis=1)

print("\n预测结果对比:")
for i in range(10):
    true_label = text_labels[y_sample[i]]
    pred_label = text_labels[int(y_pred[i])]
    result = "✓" if y_sample[i] == y_pred[i] else "✗"
    print(f"样本{i+1}: 真实={true_label:12s} 预测={pred_label:12s} {result}")

print("\n✅ 6.6 LeNet代码运行完成!")
print("\n注意: 为了节省时间，这里只训练了3个epoch。")
print("完整训练通常需要10个epoch，测试准确率可达82-83%。")
