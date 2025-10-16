#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""8.7. 通过时间反向传播 (BPTT) - 梯度消失与爆炸"""

import tensorflow as tf


print("=" * 70)
print("8.7. 通过时间反向传播 (BPTT)")
print("=" * 70)

print("""
循环神经网络通过展开时间轴来反向传播梯度。
如果每一步的雅可比矩阵特征值 < 1，梯度会趋近 0（梯度消失）；
如果 > 1，梯度会指数级增长（梯度爆炸）。

我们用一个最小化的例子演示这一现象。
""")


def gradient_through_time(a: float, time_steps: int = 20) -> float:
    """h_t = a * h_{t-1} 的简单递推，其梯度为 a^T。"""
    h0 = tf.constant(1.0)
    with tf.GradientTape() as tape:
        tape.watch(h0)
        h = h0
        for _ in range(time_steps):
            h = a * h
        loss = h
    grad = tape.gradient(loss, h0)
    return float(grad.numpy())


for a in (0.5, 0.99, 1.01, 1.5):
    grad = gradient_through_time(a)
    print(f"系数 a={a:<4} -> 经过 20 步后的梯度: {grad:.3e}")

print("""
结论：
- a=0.5/0.99 时梯度迅速衰减为 0，模型难以学习长期依赖；
- a=1.01/1.5 时梯度指数级增长，训练可能数值溢出。

梯度裁剪是防止爆炸的常用技巧。下面给出示例。
""")


def demo_gradient_clipping():
    """模拟一次梯度裁剪。"""
    grads = [tf.constant([[15.0, -8.0], [3.0, -20.0]]), tf.constant([7.0, -18.0])]
    norm = tf.linalg.global_norm(grads)
    theta = 5.0
    factor = theta / norm
    clipped = [g * factor for g in grads]
    return norm.numpy(), factor.numpy(), [g.numpy() for g in clipped]


norm, factor, clipped = demo_gradient_clipping()
print(f"\n原始梯度全局范数: {norm:.2f}")
print(f"裁剪阈值 theta=5.0 -> 缩放因子 {factor:.3f}")
print("裁剪后的梯度样例:")
for grad in clipped:
    print(grad)

print("""
在实际训练中（参考 8.5、8.6 脚本），我们会先计算 global_norm，
若超过阈值则等比例缩放所有梯度，保证稳定更新。

✅ 8.7 BPTT 示例运行完成!
""")
