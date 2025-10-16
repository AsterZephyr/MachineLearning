#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.5. 循环神经网络的从零开始实现"""

import math

import tensorflow as tf

from utils.rnn_utils import (
    Vocab,
    load_corpus_time_machine,
    seq_data_iter_random,
)


print("=" * 70)
print("8.5. 循环神经网络的从零开始实现")
print("=" * 70)

print("""
目标：不依赖 Keras 高层 API，使用张量运算手写一个字符级语言模型。
关键步骤：
1. 初始化权重
2. 编写 RNN 前向传播和梯度更新
3. 通过随机采样的小批量训练
""")

corpus, vocab = load_corpus_time_machine(max_tokens=10000, token="char")
vocab_size = len(vocab)

num_hiddens = 64
num_steps = 10
batch_size = 32
num_epochs = 20
learning_rate = 1.0
grad_clip_theta = 1.0

print(f"数据集长度: {len(corpus)}, 字符种类: {vocab_size}")
print(f"训练配置: batch_size={batch_size}, num_steps={num_steps}, num_hiddens={num_hiddens}")


def get_params(vocab_size, num_hiddens):
    """初始化权重矩阵。"""
    normal = tf.random.truncated_normal
    W_xh = tf.Variable(normal((vocab_size, num_hiddens), stddev=0.01))
    W_hh = tf.Variable(normal((num_hiddens, num_hiddens), stddev=0.01))
    b_h = tf.Variable(tf.zeros((num_hiddens,)))
    W_hq = tf.Variable(normal((num_hiddens, vocab_size), stddev=0.01))
    b_q = tf.Variable(tf.zeros((vocab_size,)))
    return [W_xh, W_hh, b_h, W_hq, b_q]


def init_state(batch_size, num_hiddens):
    return tf.zeros((batch_size, num_hiddens))


def rnn_step(X, state, params):
    """单步 RNN 计算。"""
    W_xh, W_hh, b_h, W_hq, b_q = params
    state = tf.tanh(tf.matmul(X, W_xh) + tf.matmul(state, W_hh) + b_h)
    Y = tf.matmul(state, W_hq) + b_q
    return Y, state


def rnn(inputs, state, params):
    """遍历时间维度的 RNN 前向传播。"""
    outputs = []
    for Xt in inputs:
        Yt, state = rnn_step(Xt, state, params)
        outputs.append(Yt)
    return tf.stack(outputs), state


def grad_clipping(grads, theta):
    """梯度裁剪，防止梯度爆炸。"""
    norm = tf.linalg.global_norm(grads)
    if tf.math.is_finite(norm) and norm > theta:
        grads = [g * (theta / norm) for g in grads]
    return grads


def predict(prefix: str, num_preds: int, params, vocab: Vocab):
    """使用训练好的参数生成文本。"""
    state = init_state(1, num_hiddens)
    outputs = [vocab[prefix[0]]]

    def _one_hot(idx):
        vec = tf.one_hot([idx], depth=vocab_size)
        return tf.reshape(vec, (1, vocab_size))

    # 处理前缀
    Y, state = rnn_step(_one_hot(outputs[0]), state, params)
    for ch in prefix[1:]:
        idx = vocab[ch]
        outputs.append(idx)
        Y, state = rnn_step(_one_hot(idx), state, params)

    last = outputs[-1]
    for _ in range(num_preds):
        Y, state = rnn_step(_one_hot(last), state, params)
        last = int(tf.argmax(Y, axis=1).numpy()[0])
        outputs.append(last)
    return "".join(vocab.to_tokens(outputs))


params = get_params(vocab_size, num_hiddens)

for epoch in range(num_epochs):
    data_iter = seq_data_iter_random(corpus, batch_size, num_steps)
    state = None
    total_loss = 0.0
    total_tokens = 0

    for X, Y in data_iter:
        if state is None or state.shape[0] != X.shape[0]:
            state = init_state(X.shape[0], num_hiddens)
        else:
            state = tf.stop_gradient(state)

        inputs = tf.one_hot(tf.transpose(X), depth=vocab_size)
        targets = tf.reshape(tf.transpose(Y), (-1,))

        with tf.GradientTape() as tape:
            outputs, state = rnn(inputs, state, params)
            outputs = tf.reshape(outputs, (-1, vocab_size))
            loss = tf.reduce_mean(
                tf.keras.losses.sparse_categorical_crossentropy(
                    y_true=targets,
                    y_pred=outputs,
                    from_logits=True,
                )
            )

        grads = tape.gradient(loss, params)
        grads = grad_clipping(grads, grad_clip_theta)

        for param, grad in zip(params, grads):
            param.assign_sub(learning_rate * grad)

        token_count = targets.shape[0]
        total_loss += float(loss) * token_count
        total_tokens += token_count

    ppl = math.exp(total_loss / total_tokens)
    sample = predict("time traveller ", 30, params, vocab)
    print(f"Epoch {epoch + 1:02d} | Perplexity {ppl:.2f} | Sample: {sample}")

print("\n✅ 8.5 手写 RNN 训练完成!\n")
