#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.3. 语言模型与数据集 - 用 n 元语法近似文本概率"""

import math
import random
from collections import Counter, defaultdict

from utils.rnn_utils import load_corpus_time_machine


print("=" * 70)
print("8.3. 语言模型与数据集 - 用 n 元语法近似文本概率")
print("=" * 70)

print("""
语言模型要回答的问题是：给定前面的 token，后一个 token 出现的概率是多少？
我们使用字符级《时间机器》数据，构建 1-gram 与 2-gram 统计并生成新文本。
""")

# 只取前 5000 个字符，便于快速演示
corpus, vocab = load_corpus_time_machine(max_tokens=5000, token="char")
tokens = vocab.to_tokens(corpus)

unigram_counts = Counter(tokens)
bigram_counts = Counter(zip(tokens[:-1], tokens[1:]))

print(f"样本字符数: {len(tokens)}")
print(f"不同字符种类: {len(vocab)}")
print("\nTop 10 Unigram:")
for token, freq in unigram_counts.most_common(10):
    print(f"'{token}': {freq}")

print("\nTop 10 Bigram:")
for (t1, t2), freq in bigram_counts.most_common(10):
    print(f"'{t1}{t2}': {freq}")

bigram_totals = defaultdict(int)
for (c1, _), freq in bigram_counts.items():
    bigram_totals[c1] += freq

bigram_prob = defaultdict(list)
for (c1, c2), freq in bigram_counts.items():
    total = bigram_totals[c1]
    bigram_prob[c1].append((c2, freq / total))


def generate_bigram(prefix: str, num_preds: int = 50) -> str:
    """根据二元语法概率生成字符序列。"""
    output = list(prefix)
    for _ in range(num_preds):
        prev = output[-1]
        candidates = bigram_prob.get(prev)
        if not candidates:
            # 没有统计数据时回退到最常见字符
            next_char = unigram_counts.most_common(1)[0][0]
        else:
            chars, probs = zip(*candidates)
            next_char = random.choices(chars, weights=probs, k=1)[0]
        output.append(next_char)
    return "".join(output)


sample_text = generate_bigram(prefix="t")
print("\n基于 bigram 的样例生成 (起始字符 't'):\n")
print(sample_text)


def perplexity(sequence: str) -> float:
    """计算二元语法在给定序列上的困惑度。"""
    eps = 1e-7
    log_sum = 0.0
    count = 0
    prev = sequence[0]
    for ch in sequence[1:]:
        candidates = dict(bigram_prob.get(prev, []))
        prob = candidates.get(ch, eps)
        log_sum += math.log(prob)
        count += 1
        prev = ch
    return math.exp(-log_sum / max(count, 1))


eval_segment = "".join(tokens[:200])
pp = perplexity(eval_segment)
print(f"\n在 200 字符片段上的困惑度 (Perplexity): {pp:.2f}")
print("困惑度越低，说明模型越能准确预测真实序列。\n")

print("✅ 8.3 语言模型示例运行完成!\n")
