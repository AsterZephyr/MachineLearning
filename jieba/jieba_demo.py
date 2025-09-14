#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
jieba分词工具Demo
展示jieba的主要功能：分词、词性标注、关键词提取
"""

import jieba
import jieba.posseg as pseg
import jieba.analyse


def basic_segmentation_demo():
    """基本分词功能演示"""
    print("=" * 50)
    print("1. 基本分词功能演示")
    print("=" * 50)
    
    text = "中国科学技术大学位于安徽省合肥市，是一所研究型大学。人工智能和机器学习是当前的热门技术。"
    
    seg_list = jieba.cut(text, cut_all=False)
    print(f"原文: {text}")
    print(f"精确模式: {' / '.join(seg_list)}")
    
    seg_list = jieba.cut(text, cut_all=True)
    print(f"全模式: {' / '.join(seg_list)}")
    
    seg_list = jieba.cut_for_search(text)
    print(f"搜索引擎模式: {' / '.join(seg_list)}")


def pos_tagging_demo():
    """词性标注功能演示"""
    print("\n" + "=" * 50)
    print("2. 词性标注功能演示")
    print("=" * 50)
    
    text = "我爱北京天安门，天安门上太阳升。伟大的中华人民共和国万岁！"
    
    words = pseg.cut(text)
    print(f"原文: {text}")
    print("词性标注结果:")
    for word, flag in words:
        print(f"  {word} / {flag}")
    
    print("\n常见词性标记说明:")
    pos_tags = {
        'n': '名词', 'v': '动词', 'a': '形容词', 'r': '代词',
        'd': '副词', 'p': '介词', 'c': '连词', 'u': '助词',
        'x': '标点符号', 'ns': '地名', 'nr': '人名', 'nt': '机构名'
    }
    for tag, desc in pos_tags.items():
        print(f"  {tag}: {desc}")


def keyword_extraction_demo():
    """关键词提取功能演示"""
    print("\n" + "=" * 50)
    print("3. 关键词提取功能演示")
    print("=" * 50)
    
    text = """
    人工智能（Artificial Intelligence，AI）是计算机科学的一个分支，它企图了解智能的实质，
    并生产出一种新的能以人类智能相似的方式做出反应的智能机器。该领域的研究包括机器人、
    语言识别、图像识别、自然语言处理和专家系统等。人工智能可以对人的意识、思维的信息过程
    的模拟。人工智能不是人的智能，但能像人那样思考、也可能超过人的智能。近年来，深度学习、
    机器学习、神经网络等技术快速发展，推动了人工智能在各个领域的应用。
    """
    
    print(f"原文: {text.strip()}")
    
    print("\nTF-IDF关键词提取:")
    keywords_tfidf = jieba.analyse.extract_tags(text, topK=10, withWeight=True)
    for keyword, weight in keywords_tfidf:
        print(f"  {keyword}: {weight:.4f}")
    
    print("\nTextRank关键词提取:")
    keywords_textrank = jieba.analyse.textrank(text, topK=10, withWeight=True)
    for keyword, weight in keywords_textrank:
        print(f"  {keyword}: {weight:.4f}")


def user_dict_demo():
    """用户自定义词典演示"""
    print("\n" + "=" * 50)
    print("4. 用户自定义词典演示")
    print("=" * 50)
    
    text = "李小福是创新办主任也是云计算方面的专家"
    
    print("添加自定义词汇前:")
    seg_list = jieba.cut(text, cut_all=False)
    print(f"  {' / '.join(seg_list)}")
    
    jieba.add_word('李小福')
    jieba.add_word('创新办')
    jieba.add_word('云计算')
    
    print("添加自定义词汇后:")
    seg_list = jieba.cut(text, cut_all=False)
    print(f"  {' / '.join(seg_list)}")
    
    jieba.suggest_freq('中华人民共和国', True)
    jieba.suggest_freq('台中', True)
    
    text2 = "中华人民共和国是一个伟大的国家"
    print(f"\n调整词频后: {text2}")
    seg_list = jieba.cut(text2, cut_all=False)
    print(f"  {' / '.join(seg_list)}")


def performance_demo():
    """性能测试演示"""
    print("\n" + "=" * 50)
    print("5. 性能测试演示")
    print("=" * 50)
    
    import time
    
    long_text = "人工智能技术的发展正在改变我们的生活方式。" * 1000
    
    start_time = time.time()
    list(jieba.cut(long_text))
    end_time = time.time()
    
    print(f"处理 {len(long_text)} 个字符的文本")
    print(f"分词耗时: {end_time - start_time:.4f} 秒")
    
    jieba.enable_parallel(4)
    start_time = time.time()
    list(jieba.cut(long_text))
    end_time = time.time()
    print(f"并行分词耗时: {end_time - start_time:.4f} 秒")
    jieba.disable_parallel()


def main():
    """主函数"""
    print("jieba分词工具功能演示")
    print("=" * 50)
    
    basic_segmentation_demo()
    pos_tagging_demo()
    keyword_extraction_demo()
    user_dict_demo()
    performance_demo()
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("主要功能总结:")
    print("1. 支持三种分词模式：精确模式、全模式、搜索引擎模式")
    print("2. 提供词性标注功能")
    print("3. 支持TF-IDF和TextRank关键词提取")
    print("4. 可自定义词典和调整词频")
    print("5. 支持并行分词提高性能")


if __name__ == "__main__":
    main()