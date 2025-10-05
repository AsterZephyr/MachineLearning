#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
24小时预算花费曲线对比图
"""

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_budget_chart():
    """创建24小时预算花费曲线对比图"""
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 理想花费数据 (线性增长)
    ideal_x = [0, 24]
    ideal_y = [0, 100]
    
    # 无调控花费数据
    uncontrolled_x = [0, 2, 4, 6, 24]
    uncontrolled_y = [0, 50, 80, 100, 100]
    
    # PID调控花费数据
    pid_x = [0, 4, 8, 12, 16, 20, 24]
    pid_y = [0, 18, 35, 50, 68, 85, 100]
    
    # 绘制三条曲线
    ax.plot(ideal_x, ideal_y, '--', linewidth=2, color='green', label='理想花费', alpha=0.8)
    ax.plot(uncontrolled_x, uncontrolled_y, '-', linewidth=2, color='red', label='无调控花费', alpha=0.8)
    ax.plot(pid_x, pid_y, '-', linewidth=2, color='blue', label='PID调控花费', alpha=0.8)
    
    # 设置坐标轴
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 100)
    
    # 设置标题和标签
    ax.set_title('24小时预算花费曲线对比', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('时间 (小时)', fontsize=12)
    ax.set_ylabel('累计花费 (%)', fontsize=12)
    
    # 设置网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    
    # 设置x轴刻度
    ax.set_xticks(range(0, 25, 2))
    
    # 设置y轴刻度
    ax.set_yticks(range(0, 101, 10))
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=11)
    
    # 添加数据点标记
    ax.scatter(uncontrolled_x, uncontrolled_y, color='red', s=50, alpha=0.7, zorder=5)
    ax.scatter(pid_x, pid_y, color='blue', s=50, alpha=0.7, zorder=5)
    ax.scatter(ideal_x, ideal_y, color='green', s=50, alpha=0.7, zorder=5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('budget_chart.png', dpi=300, bbox_inches='tight')
    plt.savefig('budget_chart.pdf', bbox_inches='tight')
    
    # 显示图表
    plt.show()
    
    print("图表已生成并保存为 budget_chart.png 和 budget_chart.pdf")

if __name__ == "__main__":
    create_budget_chart()
