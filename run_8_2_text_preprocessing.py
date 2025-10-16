#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.2. 文本预处理 - 从原始字符到索引序列"""

import itertools
from collections import Counter

from utils.rnn_utils import (
    Vocab,
    load_corpus_time_machine,
    read_time_machine,
    seq_data_iter_random,
    seq_data_iter_sequential,
    tokenize,
)


print("=" * 70)
print("8.2. 文本预处理 - 从原始字符到索引序列")
print("=" * 70)

print("""
《时间机器》文本将作为后续章节的语言建模数据集。
核心流程：
1. 清洗文本（只保留字母）
2. 按词/字符切分
3. 构建词表，将 token 映射为整数索引
4. 生成可供小批量训练的定长序列
""")

lines = read_time_machine()[:5]
print("样例原始句子（清洗后）:")
for line in lines:
    print(f"- {line}")

word_tokens = tokenize(lines, token="word")
char_tokens = tokenize(lines, token="char")

print("\n按词切分后的前两个句子:")
for tokens in word_tokens[:2]:
    print(tokens)

print("\n按字符切分后的前两个句子:")
for tokens in char_tokens[:2]:
    print(tokens[:30], "...")

vocab_word = Vocab(word_tokens)
vocab_char = Vocab(char_tokens)

print(f"\n词表大小（word-level）: {len(vocab_word)}")
print(f"词表大小（char-level）: {len(vocab_char)}")

top_words = Counter(itertools.chain.from_iterable(word_tokens)).most_common(5)
print("\n高频词示例:", top_words)

sample_line = word_tokens[0][:5]
sample_ids = vocab_word[sample_line]
print(f"\n示例词序列: {sample_line}")
print(f"对应的索引序列: {sample_ids}")

corpus, vocab = load_corpus_time_machine(max_tokens=2000, token="char")
print(f"\n读取 corpus（char-level）长度: {len(corpus)}")
print("前30个字符索引:", corpus[:30])
print("映射回字符:", "".join(vocab.to_tokens(corpus[:30])))


def preview_iterator(iterator, name: str):
    """打印迭代器输出的第一个小批量。"""
    print(f"\n{name} - 取一个小批量 (batch_size=2, num_steps=5):")
    for X, Y in iterator:
        print("X:", X.numpy())
        print("Y:", Y.numpy())
        break


preview_iterator(
    seq_data_iter_random(corpus, batch_size=2, num_steps=5),
    "随机抽样 (seq_data_iter_random)",
)

preview_iterator(
    seq_data_iter_sequential(corpus, batch_size=2, num_steps=5),
    "顺序划分 (seq_data_iter_sequential)",
)

print("\n✅ 8.2 文本预处理示例运行完成!\n")
