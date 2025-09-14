#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Word2Vec词向量模型Demo
展示Word2Vec的主要功能：词向量训练、相似词查找、词向量运算
"""

import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import jieba
import warnings
warnings.filterwarnings('ignore')


def prepare_training_data():
    """准备训练数据"""
    print("=" * 50)
    print("1. 准备训练数据")
    print("=" * 50)
    
    corpus = [
        "我爱中国，中国是我的祖国",
        "北京是中国的首都，也是一座美丽的城市",
        "上海是中国的经济中心，发展很快",
        "深圳是一座年轻的城市，科技发达",
        "杭州有美丽的西湖，风景如画",
        "苏州园林闻名世界，古典优雅",
        "成都的火锅很有名，麻辣鲜香",
        "西安有兵马俑，历史悠久",
        "广州是南方的大城市，经济发达",
        "天津靠近北京，交通便利",
        "人工智能是未来的发展方向",
        "机器学习需要大量的数据训练",
        "深度学习在图像识别方面很强",
        "自然语言处理是AI的重要分支",
        "神经网络模拟人脑的工作方式",
        "算法是计算机科学的核心",
        "大数据分析帮助企业决策",
        "云计算提供了便利的服务",
        "区块链技术保证数据安全",
        "物联网连接万物互联"
    ]
    
    segmented_sentences = []
    for sentence in corpus:
        words = list(jieba.cut(sentence))
        words = [word for word in words if len(word) > 1 and word.isalnum()]
        if words:
            segmented_sentences.append(words)
    
    print(f"语料库大小: {len(corpus)} 句")
    print("分词示例:")
    for i, sentence in enumerate(segmented_sentences[:3]):
        print(f"  句子{i+1}: {' / '.join(sentence)}")
    
    return segmented_sentences


def train_word2vec_model(sentences):
    """训练Word2Vec模型"""
    print("\n" + "=" * 50)
    print("2. 训练Word2Vec模型")
    print("=" * 50)
    
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,
        window=5,
        min_count=1,
        workers=4,
        sg=1,
        epochs=100
    )
    
    print(f"词汇表大小: {len(model.wv.key_to_index)}")
    print(f"词向量维度: {model.wv.vector_size}")
    print("词汇表前20个词:")
    vocab_words = list(model.wv.key_to_index.keys())[:20]
    print(f"  {vocab_words}")
    
    return model


def word_similarity_demo(model):
    """词相似度演示"""
    print("\n" + "=" * 50)
    print("3. 词相似度计算")
    print("=" * 50)
    
    test_pairs = [
        ("北京", "上海"),
        ("中国", "祖国"),
        ("城市", "发展"),
        ("人工智能", "机器学习"),
        ("数据", "算法")
    ]
    
    print("词对相似度:")
    for word1, word2 in test_pairs:
        try:
            similarity = model.wv.similarity(word1, word2)
            print(f"  {word1} - {word2}: {similarity:.4f}")
        except KeyError:
            print(f"  {word1} - {word2}: 词不在词汇表中")


def most_similar_demo(model):
    """相似词查找演示"""
    print("\n" + "=" * 50)
    print("4. 相似词查找")
    print("=" * 50)
    
    test_words = ["中国", "城市", "人工智能", "算法"]
    
    for word in test_words:
        try:
            similar_words = model.wv.most_similar(word, topn=5)
            print(f"\n与 '{word}' 最相似的词:")
            for similar_word, score in similar_words:
                print(f"  {similar_word}: {score:.4f}")
        except KeyError:
            print(f"  词 '{word}' 不在词汇表中")


def word_analogy_demo(model):
    """词类比演示"""
    print("\n" + "=" * 50)
    print("5. 词类比 (词向量运算)")
    print("=" * 50)
    
    try:
        print("尝试词类比运算 (由于语料较小，结果可能不够理想):")
        
        words_to_check = ["北京", "中国", "上海"]
        available_words = []
        for word in words_to_check:
            if word in model.wv.key_to_index:
                available_words.append(word)
        
        if len(available_words) >= 3:
            result = model.wv.most_similar(
                positive=[available_words[0], available_words[2]], 
                negative=[available_words[1]], 
                topn=3
            )
            print(f"  {available_words[0]} - {available_words[1]} + {available_words[2]} ≈")
            for word, score in result:
                print(f"    {word}: {score:.4f}")
        else:
            print("  语料中词汇不足，无法进行类比运算")
            
    except Exception as e:
        print(f"  类比运算失败: {e}")


def vector_visualization_demo(model):
    """词向量可视化演示"""
    print("\n" + "=" * 50)
    print("6. 词向量可视化")
    print("=" * 50)
    
    words = list(model.wv.key_to_index.keys())[:15]
    
    word_vectors = np.array([model.wv[word] for word in words])
    
    pca = PCA(n_components=2)
    word_vectors_2d = pca.fit_transform(word_vectors)
    
    plt.figure(figsize=(12, 8))
    plt.scatter(word_vectors_2d[:, 0], word_vectors_2d[:, 1], alpha=0.7)
    
    for i, word in enumerate(words):
        plt.annotate(word, (word_vectors_2d[i, 0], word_vectors_2d[i, 1]), 
                    fontsize=12, alpha=0.8)
    
    plt.title('Word2Vec词向量可视化 (PCA降维)', fontsize=14)
    plt.xlabel('第一主成分')
    plt.ylabel('第二主成分')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/Users/hxz/code/MachineLearning/word2vec/word_vectors_visualization.png', 
                dpi=300, bbox_inches='tight')
    print("词向量可视化图已保存为 'word_vectors_visualization.png'")
    plt.close()


def save_and_load_model_demo(model):
    """模型保存和加载演示"""
    print("\n" + "=" * 50)
    print("7. 模型保存和加载")
    print("=" * 50)
    
    model_path = '/Users/hxz/code/MachineLearning/word2vec/word2vec_model.model'
    model.save(model_path)
    print(f"模型已保存到: {model_path}")
    
    loaded_model = Word2Vec.load(model_path)
    print("模型加载成功")
    
    print("验证加载的模型:")
    if "中国" in loaded_model.wv.key_to_index:
        similar_words = loaded_model.wv.most_similar("中国", topn=3)
        print(f"  与'中国'最相似的词: {similar_words}")
    
    vector_path = '/Users/hxz/code/MachineLearning/word2vec/word_vectors.kv'
    model.wv.save(vector_path)
    print(f"词向量已保存到: {vector_path}")


def model_evaluation_demo(model):
    """模型评估演示"""
    print("\n" + "=" * 50)
    print("8. 模型评估")
    print("=" * 50)
    
    vocab_size = len(model.wv.key_to_index)
    print(f"词汇表大小: {vocab_size}")
    
    all_vectors = np.array([model.wv[word] for word in model.wv.key_to_index])
    
    print(f"向量维度: {model.wv.vector_size}")
    print(f"向量均值: {np.mean(all_vectors):.4f}")
    print(f"向量标准差: {np.std(all_vectors):.4f}")
    
    norms = [np.linalg.norm(model.wv[word]) for word in list(model.wv.key_to_index.keys())[:10]]
    print(f"前10个词的向量范数: {[f'{norm:.2f}' for norm in norms]}")


def main():
    """主函数"""
    print("Word2Vec词向量模型功能演示")
    print("=" * 50)
    
    sentences = prepare_training_data()
    
    model = train_word2vec_model(sentences)
    
    word_similarity_demo(model)
    most_similar_demo(model)
    word_analogy_demo(model)
    vector_visualization_demo(model)
    save_and_load_model_demo(model)
    model_evaluation_demo(model)
    
    print("\n" + "=" * 50)
    print("演示完成！")
    print("主要功能总结:")
    print("1. 从文本语料训练词向量模型")
    print("2. 计算词与词之间的相似度")
    print("3. 查找与给定词最相似的词")
    print("4. 进行词向量运算和类比推理")
    print("5. 词向量可视化")
    print("6. 模型保存和加载")
    print("7. 模型评估和统计")


if __name__ == "__main__":
    main()