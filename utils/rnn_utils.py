#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
循环神经网络示例的通用数据预处理工具。

提供：
1. 读取《时间机器》文本并清洗
2. 构建词表（字符级或词级）
3. 将序列切分成小批量迷你序列（随机抽样或顺序分割）

实现参考：李沐《动手学深度学习》。
"""

from __future__ import annotations

import collections
import random
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import tensorflow as tf


_DATA_DIR = Path(__file__).resolve().parent.parent / "datasets"
_TIME_MACHINE_PATH = _DATA_DIR / "timemachine.txt"


def read_time_machine() -> List[str]:
    """读取并清洗《时间机器》文本（一行一句）。"""
    with _TIME_MACHINE_PATH.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        # 只保留字母，替换其他字符为空格，并转换为小写
        line = re.sub(r"[^A-Za-z]+", " ", line).strip().lower()
        if line:
            cleaned.append(line)
    return cleaned


def tokenize(lines: Sequence[str], token: str = "word") -> List[List[str]]:
    """将文本按词或字符切分成嵌套列表。"""
    if token == "word":
        return [line.split() for line in lines]
    if token == "char":
        return [list(line) for line in lines]
    raise ValueError("token must be 'word' or 'char'")


class Vocab:
    """简单的词表封装，提供 token ↔ index 的映射。"""

    def __init__(
        self,
        tokens: Iterable[Iterable[str]],
        min_freq: int = 0,
        reserved_tokens: Iterable[str] | None = None,
    ):
        reserved_tokens = list(reserved_tokens) if reserved_tokens else []
        counter = collections.Counter(token for line in tokens for token in line)
        # 频率从高到低，相同频率按字母序
        self.token_freqs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        self.idx_to_token = ["<unk>"] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}

        for token, freq in self.token_freqs:
            if freq < min_freq:
                continue
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self) -> int:
        return len(self.idx_to_token)

    def __getitem__(self, tokens: str | Sequence[str]) -> int | List[int]:
        if isinstance(tokens, str):
            return self.token_to_idx.get(tokens, 0)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices: int | Sequence[int]) -> str | List[str]:
        if isinstance(indices, int):
            return self.idx_to_token[indices]
        return [self.idx_to_token[i] for i in indices]


def load_corpus_time_machine(
    max_tokens: int | None = None,
    token: str = "char",
) -> Tuple[List[int], Vocab]:
    """
    返回《时间机器》的索引序列以及词表。

    Args:
        max_tokens: 限制返回的最大 token 数，None 表示不限制。
        token: 'char' 或 'word'。
    """
    lines = read_time_machine()
    tokens = tokenize(lines, token)
    vocab = Vocab(tokens)

    corpus: List[int] = []
    for line in tokens:
        corpus.extend(vocab[line])

    if max_tokens is not None:
        corpus = corpus[: max_tokens]
    return corpus, vocab


def seq_data_iter_random(
    corpus: Sequence[int],
    batch_size: int,
    num_steps: int,
) -> Iterable[Tuple[tf.Tensor, tf.Tensor]]:
    """
    随机采样小批量序列。

    每个批次返回形状为 (batch_size, num_steps) 的 X、Y。
    """
    corpus = corpus[random.randint(0, num_steps - 1) :]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def _data(pos: int) -> List[int]:
        return corpus[pos : pos + num_steps]

    for i in range(0, len(initial_indices), batch_size):
        batch_indices = initial_indices[i : i + batch_size]
        X = [_data(j) for j in batch_indices]
        Y = [_data(j + 1) for j in batch_indices]
        yield tf.constant(X, dtype=tf.int32), tf.constant(Y, dtype=tf.int32)


def seq_data_iter_sequential(
    corpus: Sequence[int],
    batch_size: int,
    num_steps: int,
) -> Iterable[Tuple[tf.Tensor, tf.Tensor]]:
    """
    顺序划分小批量序列。

    保证同一小批量中不同样本是连续片段，便于在训练中沿时间维传递隐藏状态。
    """
    offset = random.randint(0, num_steps)
    num_tokens = ((len(corpus) - offset - 1) // batch_size) * batch_size
    Xs = tf.constant(corpus[offset : offset + num_tokens], dtype=tf.int32)
    Ys = tf.constant(corpus[offset + 1 : offset + 1 + num_tokens], dtype=tf.int32)
    Xs = tf.reshape(Xs, (batch_size, -1))
    Ys = tf.reshape(Ys, (batch_size, -1))

    num_batches = Xs.shape[1] // num_steps
    for i in range(0, num_batches * num_steps, num_steps):
        X = Xs[:, i : i + num_steps]
        Y = Ys[:, i : i + num_steps]
        yield X, Y


def get_device() -> str:
    """方便脚本打印当前运行的设备信息。"""
    gpus = tf.config.list_logical_devices("GPU")
    if gpus:
        return gpus[0].name
    return "CPU"
