"""
简单的d2l替代模块，提供基本的绘图功能
"""
import matplotlib.pyplot as plt
import numpy as np

def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None, ylim=None,
         xscale='linear', yscale='linear', fmts=('-', 'm--', 'g-.', 'r:'),
         figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []
    
    plt.rcParams['figure.figsize'] = figsize
    axes = axes if axes else plt.gca()
    
    # 如果X只有一个维度，将其转换为列向量
    if hasattr(X, 'ndim') and X.ndim == 1:
        X = X.reshape(-1, 1)
    
    # 如果Y是None，则X是x轴数据，Y是y轴数据
    if Y is None:
        X, Y = X[:, 0], X[:, 1] if X.shape[1] > 1 else X[:, 0]
    
    # 确保X和Y是numpy数组
    if not isinstance(X, np.ndarray):
        X = np.array(X)
    if not isinstance(Y, np.ndarray):
        Y = np.array(Y)
    
    # 如果Y是二维的，绘制多条线
    if Y.ndim > 1 and Y.shape[0] > 1:
        for i, (y, fmt) in enumerate(zip(Y, fmts)):
            axes.plot(X, y, fmt, label=legend[i] if i < len(legend) else f'line {i}')
    else:
        # 单条线
        fmt = fmts[0] if isinstance(fmts, (list, tuple)) else fmts
        axes.plot(X, Y, fmt, label=legend[0] if legend else None)
    
    if xlabel:
        axes.set_xlabel(xlabel)
    if ylabel:
        axes.set_ylabel(ylabel)
    if legend:
        axes.legend()
    if xlim:
        axes.set_xlim(xlim)
    if ylim:
        axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    
    plt.show()

def set_figsize(figsize=(3.5, 2.5)):
    """设置图形大小"""
    plt.rcParams['figure.figsize'] = figsize

def use_svg_display():
    """使用SVG显示"""
    plt.rcParams['figure.figsize'] = (3.5, 2.5)
    plt.rcParams['font.size'] = 10
