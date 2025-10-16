#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.6. 循环神经网络的简洁实现 (TensorFlow/Keras)"""

import math

import numpy as np
import tensorflow as tf

from utils.rnn_utils import load_corpus_time_machine


print("=" * 70)
print("8.6. 循环神经网络的简洁实现 (TensorFlow/Keras)")
print("=" * 70)

print("""
相比 8.5 的“从零开始”，这里使用 Keras 高层 API 快速搭建训练流程：
1. Embedding 层将字符索引映射到向量
2. SimpleRNN 层学习时间依赖
3. Dense 层输出下一个字符的 logits
""")

corpus, vocab = load_corpus_time_machine(max_tokens=6000, token="char")
vocab_size = len(vocab)
num_steps = 20
batch_size = 32
embedding_dim = 64
rnn_units = 128
epochs = 5


def build_dataset(corpus, num_steps, batch_size):
    """将连续序列切成样本-标签对并装入 tf.data.Dataset。"""
    Xs, Ys = [], []
    for i in range(len(corpus) - num_steps):
        Xs.append(corpus[i : i + num_steps])
        Ys.append(corpus[i + 1 : i + 1 + num_steps])
    X = np.array(Xs, dtype=np.int32)
    Y = np.array(Ys, dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.shuffle(2048).batch(batch_size, drop_remainder=True)
    return dataset


dataset = build_dataset(corpus, num_steps, batch_size)

inputs = tf.keras.Input(shape=(None,), dtype=tf.int32)
embedding_layer = tf.keras.layers.Embedding(vocab_size, embedding_dim, name="embedding")
rnn_layer = tf.keras.layers.SimpleRNN(
    rnn_units,
    return_sequences=True,
    name="rnn",
)
dense_layer = tf.keras.layers.Dense(vocab_size, name="dense")

x = embedding_layer(inputs)
x = rnn_layer(x)
logits = dense_layer(x)

model = tf.keras.Model(inputs=inputs, outputs=logits)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
)

history = model.fit(dataset, epochs=epochs, verbose=2)

loss = model.evaluate(dataset, verbose=0)
perplexity = math.exp(loss)
print(f"\n训练完成 - 最终损失: {loss:.3f}, 困惑度: {perplexity:.2f}")


def generate_text(prefix: str, num_preds: int = 40) -> str:
    """直接复用训练好的权重逐字符生成文本。"""
    embedding_w = embedding_layer.get_weights()[0]
    rnn_kernel, rnn_recurrent, rnn_bias = rnn_layer.get_weights()
    dense_w, dense_b = dense_layer.get_weights()

    state = tf.zeros((1, rnn_units))
    outputs = [vocab[prefix[0]]]

    def _embed(idx):
        vec = tf.nn.embedding_lookup(embedding_w, [idx])
        return tf.reshape(vec, (1, embedding_dim))

    def _step(idx, state):
        x = _embed(idx)
        state = tf.tanh(tf.matmul(x, rnn_kernel) + tf.matmul(state, rnn_recurrent) + rnn_bias)
        logits = tf.matmul(state, dense_w) + dense_b
        return logits, state

    _, state = _step(outputs[0], state)
    for ch in prefix[1:]:
        idx = vocab[ch]
        outputs.append(idx)
        _, state = _step(idx, state)

    last = outputs[-1]
    for _ in range(num_preds):
        logits, state = _step(last, state)
        last = int(tf.argmax(logits, axis=-1).numpy()[0])
        outputs.append(last)
    return "".join(vocab.to_tokens(outputs))


sample = generate_text("time traveller ", num_preds=50)
print("\n基于训练模型的生成文本:")
print(sample)

print("\n✅ 8.6 Keras 简洁实现运行完成!\n")
