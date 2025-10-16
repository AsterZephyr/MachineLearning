#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.1. 序列模型 - 为什么需要记忆?"""

from typing import List
import tensorflow as tf

from utils.rnn_utils import get_device


print("=" * 70)
print("8.1. 序列模型 - 为什么需要记忆?")
print("=" * 70)

print("""
序列数据（语音、文本、股价...）的核心特征是【顺序】。
如果我们像处理独立同分布 (i.i.d.) 数据那样忽略顺序，模型就无法捕捉上下文关联。

本节通过两个小例子说明：
1. 语言模型中，字符之间存在强烈的条件依赖。
2. 循环神经网络 (RNN) 通过隐藏状态携带“记忆”。
""")


def independent_prob(sequence: List[str], probs: dict) -> float:
    """基于独立假设计算序列的联合概率。"""
    p = 1.0
    for token in sequence:
        p *= probs.get(token, 1e-6)
    return p


def bigram_prob(sequence: List[str], bigram_probs: dict, start_token="<bos>") -> float:
    """基于二元语法 (bigram) 的链式法则概率。"""
    p = 1.0
    prev = start_token
    for token in sequence:
        p *= bigram_probs.get((prev, token), 1e-6)
        prev = token
    return p


word = list("time")

unigram = {"t": 0.05, "i": 0.02, "m": 0.01, "e": 0.06}
bigram = {
    ("<bos>", "t"): 0.40,
    ("t", "i"): 0.65,
    ("i", "m"): 0.55,
    ("m", "e"): 0.72,
}

p_independent = independent_prob(word, unigram)
p_bigram = bigram_prob(word, bigram)

print("示例：预测单词 'time'")
print(f"- 独立假设联合概率: {p_independent:.2e}")
print(f"- 二元语法联合概率: {p_bigram:.2e}")
print("""
Bigram 通过“记住”前一个字符，大幅提升概率估计。
这说明记忆历史信息对于建模序列至关重要。
""")


print("=" * 70)
print("循环神经网络如何携带记忆？")
print("=" * 70)

vocab = {"<pad>": 0, "t": 1, "i": 2, "m": 3, "e": 4}
sequence = [vocab[c] for c in word]

embedding = tf.keras.layers.Embedding(input_dim=len(vocab), output_dim=4)
rnn_cell = tf.keras.layers.SimpleRNNCell(units=3, activation="tanh")

state = tf.zeros((1, 3))

print(f"运行设备: {get_device()}")
print("\n逐步输入字符，观察隐藏状态（记忆向量）的演化：")

for step, token_id in enumerate(sequence):
    x = embedding(tf.constant([token_id]))  # (1, 4)
    output, [state] = rnn_cell(x, [state])
    decoded = ", ".join(f"{value: .3f}" for value in tf.squeeze(state).numpy())
    print(f"Step {step + 1} 输入 '{word[step]}' -> 隐藏状态: [{decoded}]")

print("""
在 RNN 中：
- 每一个时间步都会更新隐藏状态 h_t = f(x_t, h_{t-1})
- h_t 对后续时间步可见，相当于“记住”前面的信息

因此，RNN 能够理解序列中的上下文依赖关系，是处理序列数据的基本模型。
""")

print("✅ 8.1 序列模型示例运行完毕!\n")
