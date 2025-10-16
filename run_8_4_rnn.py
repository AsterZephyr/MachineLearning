#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.4. 循环神经网络 - 手写前向传播"""

import tensorflow as tf

from utils.rnn_utils import (
    load_corpus_time_machine,
    seq_data_iter_sequential,
)


print("=" * 70)
print("8.4. 循环神经网络 - 手写前向传播")
print("=" * 70)

print("""
循环神经网络的核心公式：
    h_t = tanh(X_t W_xh + h_{t-1} W_hh + b_h)
    y_t = h_t W_hq + b_q

我们用字符级《时间机器》数据构造一个迷你批次，
以纯 TensorFlow 张量运算实现一次前向传播，帮助理解 RNN 内部结构。
""")

corpus, vocab = load_corpus_time_machine(max_tokens=1000, token="char")
num_hiddens = 16
batch_size, num_steps = 2, 5

iterator = seq_data_iter_sequential(corpus, batch_size, num_steps)
X_batch, Y_batch = next(iter(iterator))

def get_params(vocab_size, num_hiddens):
    """初始化权重矩阵。"""
    normal = tf.random.normal
    W_xh = tf.Variable(normal((vocab_size, num_hiddens), stddev=0.01))
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens), stddev=0.01))
    b_h = tf.Variable(tf.zeros((num_hiddens,)))
    W_hq = tf.Variable(normal((num_hiddens, vocab_size), stddev=0.01))
    b_q = tf.Variable(tf.zeros((vocab_size,)))
    return W_xh, W_hh, b_h, W_hq, b_q


def init_rnn_state(batch_size, num_hiddens):
    """隐藏状态初始化为全零。"""
    return tf.zeros((batch_size, num_hiddens))


def rnn(inputs, state, params):
    """手写 RNN 前向传播。"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    outputs = []
    for Xt in inputs:  # Xt: (batch_size, vocab_size)
        state = tf.tanh(tf.matmul(Xt, W_xh) + tf.matmul(state, W_hh) + b_h)
        Yt = tf.matmul(state, W_hq) + b_q
        outputs.append(Yt)
    return tf.stack(outputs), state


params = get_params(len(vocab), num_hiddens)
state = init_rnn_state(batch_size, num_hiddens)

# 将索引序列 one-hot 化供矩阵乘法使用：(num_steps, batch_size, vocab_size)
inputs = tf.one_hot(tf.transpose(X_batch), depth=len(vocab))

outputs, final_state = rnn(inputs, state, params)

print(f"输入 X_batch 形状: {X_batch.shape}")
print(f"输入 one-hot 序列形状: {inputs.shape}")
print(f"输出 logits 形状: {outputs.shape} (time, batch, vocab)")
print(f"最终隐藏状态形状: {final_state.shape}")

print("\ntime step 1 的输出 logits（截断三列）:")
print(outputs[0][:, :3])

print("""
可以看到：
- 隐藏状态在时间维度上递推，携带上下文信息；
- 每个时间步的输出都依赖于当前输入与上一隐藏状态。

实际训练时只需加上损失函数与梯度更新即可。
""")

print("✅ 8.4 RNN 前向传播示例运行完成!\n")
