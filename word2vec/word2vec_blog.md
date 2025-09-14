# 深入理解Word2Vec：从词向量到语义表示的完整解析

## 引言

在自然语言处理（NLP）领域，如何将词汇转换为计算机能够理解的数值表示是一个根本性问题。Word2Vec作为里程碑式的词向量模型，通过神经网络学习将词汇映射到高维向量空间，使得词汇之间的语义关系能够通过向量运算来体现。本文将深入探讨Word2Vec的底层原理，帮助读者理解这个革命性模型是如何工作的。

## Word2Vec的核心思想

### 1. 分布式假设

Word2Vec基于一个核心假设：**出现在相似上下文中的词汇具有相似的语义**。这意味着：

```
相似的词汇 → 相似的上下文 → 相似的向量表示
```

### 2. 向量空间表示

每个词汇被映射到一个高维向量空间中的点：

```python
# 词汇到向量的映射
word_vectors = {
    "中国": [0.1, -0.3, 0.8, ...],      # 100维向量
    "北京": [0.2, -0.1, 0.7, ...],      # 100维向量
    "上海": [0.3, -0.2, 0.6, ...],      # 100维向量
    # ... 更多词汇
}
```

## 两种训练架构详解

### 1. CBOW (Continuous Bag of Words) 模型

CBOW模型通过上下文词汇预测目标词汇：

```
输入: [w(t-2), w(t-1), w(t+1), w(t+2)] → 输出: w(t)
```

#### 网络结构

```python
class CBOWModel:
    def __init__(self, vocab_size, embedding_dim, context_size):
        self.vocab_size = vocab_size          # 词汇表大小
        self.embedding_dim = embedding_dim    # 词向量维度
        self.context_size = context_size      # 上下文窗口大小
        
        # 输入层到隐藏层的权重矩阵（词向量矩阵）
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # 隐藏层到输出层的权重矩阵
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        
        # 偏置项
        self.b1 = np.zeros(embedding_dim)
        self.b2 = np.zeros(vocab_size)
    
    def forward(self, context_indices):
        """前向传播"""
        # 1. 将上下文词汇转换为one-hot向量
        context_vectors = np.zeros((len(context_indices), self.vocab_size))
        for i, idx in enumerate(context_indices):
            context_vectors[i, idx] = 1
        
        # 2. 计算隐藏层（平均上下文向量）
        hidden = np.mean(context_vectors @ self.W1, axis=0) + self.b1
        
        # 3. 计算输出层
        output = hidden @ self.W2 + self.b2
        
        # 4. 应用softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        
        return probabilities, hidden
```

### 2. Skip-gram 模型

Skip-gram模型与CBOW相反，通过目标词汇预测上下文：

```
输入: w(t) → 输出: [w(t-2), w(t-1), w(t+1), w(t+2)]
```

#### 网络结构

```python
class SkipGramModel:
    def __init__(self, vocab_size, embedding_dim):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # 输入词向量矩阵
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        
        # 输出词向量矩阵
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        
        # 偏置项
        self.b1 = np.zeros(embedding_dim)
        self.b2 = np.zeros(vocab_size)
    
    def forward(self, input_idx):
        """前向传播"""
        # 1. 输入层（one-hot向量）
        input_vector = np.zeros(self.vocab_size)
        input_vector[input_idx] = 1
        
        # 2. 隐藏层（词向量）
        hidden = input_vector @ self.W1 + self.b1
        
        # 3. 输出层
        output = hidden @ self.W2 + self.b2
        
        # 4. 应用softmax
        exp_output = np.exp(output - np.max(output))
        probabilities = exp_output / np.sum(exp_output)
        
        return probabilities, hidden
```

## 负采样优化技术

### 1. 负采样的必要性

传统softmax计算成本高：
```python
# 传统方法：需要计算所有词汇的概率
def traditional_softmax(output, target_idx):
    exp_output = np.exp(output)
    probabilities = exp_output / np.sum(exp_output)
    return -np.log(probabilities[target_idx])

# 负采样：只计算目标词和少量负样本
def negative_sampling_loss(output, target_idx, negative_indices):
    # 正样本损失
    pos_score = output[target_idx]
    pos_loss = -np.log(1 / (1 + np.exp(-pos_score)))
    
    # 负样本损失
    neg_loss = 0
    for neg_idx in negative_indices:
        neg_score = output[neg_idx]
        neg_loss += -np.log(1 / (1 + np.exp(neg_score)))
    
    return pos_loss + neg_loss
```

### 2. 负采样实现

```python
class NegativeSamplingSkipGram:
    def __init__(self, vocab_size, embedding_dim, negative_samples=5):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.negative_samples = negative_samples
        
        # 词向量矩阵
        self.W1 = np.random.randn(vocab_size, embedding_dim) * 0.01
        self.W2 = np.random.randn(embedding_dim, vocab_size) * 0.01
        
        # 负采样概率分布（基于词频）
        self.negative_probs = self._calculate_negative_probs()
    
    def _calculate_negative_probs(self):
        """计算负采样概率分布"""
        # 这里简化处理，实际应该基于真实词频
        word_counts = np.ones(self.vocab_size)
        probs = word_counts ** 0.75
        probs = probs / np.sum(probs)
        return probs
    
    def sample_negative_words(self, target_idx):
        """采样负样本词汇"""
        negative_indices = []
        for _ in range(self.negative_samples):
            while True:
                neg_idx = np.random.choice(self.vocab_size, p=self.negative_probs)
                if neg_idx != target_idx and neg_idx not in negative_indices:
                    negative_indices.append(neg_idx)
                    break
        return negative_indices
```

## 词向量相似度计算

### 1. 余弦相似度

```python
def cosine_similarity(vec1, vec2):
    """计算两个向量的余弦相似度"""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0
    
    return dot_product / (norm1 * norm2)

def word_similarity(model, word1, word2):
    """计算两个词的相似度"""
    if word1 not in model.wv.key_to_index or word2 not in model.wv.key_to_index:
        return None
    
    vec1 = model.wv[word1]
    vec2 = model.wv[word2]
    
    return cosine_similarity(vec1, vec2)
```

### 2. 相似词查找

```python
def find_most_similar(model, target_word, top_n=10):
    """查找与目标词最相似的词汇"""
    if target_word not in model.wv.key_to_index:
        return []
    
    target_vector = model.wv[target_word]
    similarities = []
    
    for word, idx in model.wv.key_to_index.items():
        if word != target_word:
            word_vector = model.wv[word]
            similarity = cosine_similarity(target_vector, word_vector)
            similarities.append((word, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
```

## 词向量运算与类比推理

### 1. 向量运算原理

Word2Vec的神奇之处在于词向量可以进行数学运算：

```python
def word_analogy(model, positive_words, negative_words, top_n=5):
    """词类比运算"""
    # 计算目标向量
    target_vector = np.zeros(model.wv.vector_size)
    
    # 正样本贡献
    for word in positive_words:
        if word in model.wv.key_to_index:
            target_vector += model.wv[word]
    
    # 负样本贡献
    for word in negative_words:
        if word in model.wv.key_to_index:
            target_vector -= model.wv[word]
    
    # 归一化
    target_vector = target_vector / np.linalg.norm(target_vector)
    
    # 查找最相似的词
    similarities = []
    for word, idx in model.wv.key_to_index.items():
        if word not in positive_words and word not in negative_words:
            word_vector = model.wv[word]
            similarity = cosine_similarity(target_vector, word_vector)
            similarities.append((word, similarity))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_n]
```

### 2. 经典类比示例

```python
# 经典类比：king - man + woman ≈ queen
def classic_analogy_demo(model):
    """经典词类比演示"""
    analogies = [
        (["king", "woman"], ["man"]),           # king - man + woman ≈ queen
        (["paris", "germany"], ["france"]),     # paris - france + germany ≈ berlin
        (["big", "bigger"], ["small"]),         # big - small + bigger ≈ smaller
    ]
    
    for pos_words, neg_words in analogies:
        print(f"类比: {' + '.join(pos_words)} - {' + '.join(neg_words)}")
        result = word_analogy(model, pos_words, neg_words, top_n=3)
        for word, score in result:
            print(f"  → {word}: {score:.4f}")
        print()
```

## 训练优化技巧

### 1. 学习率调度

```python
class LearningRateScheduler:
    def __init__(self, initial_lr=0.025, min_lr=0.0001):
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.current_lr = initial_lr
    
    def step(self, epoch, total_epochs):
        """更新学习率"""
        if epoch < total_epochs * 0.8:
            # 前80%的epoch保持初始学习率
            self.current_lr = self.initial_lr
        else:
            # 后20%的epoch线性衰减
            progress = (epoch - total_epochs * 0.8) / (total_epochs * 0.2)
            self.current_lr = self.initial_lr * (1 - progress) + self.min_lr * progress
        
        return self.current_lr
```

### 2. 批量训练

```python
def batch_training(sentences, vocab, batch_size=32, embedding_dim=100):
    """批量训练Word2Vec模型"""
    model = SkipGramModel(len(vocab), embedding_dim)
    scheduler = LearningRateScheduler()
    
    # 准备训练数据
    training_data = []
    for sentence in sentences:
        for i in range(len(sentence)):
            target_word = sentence[i]
            context_words = []
            
            # 获取上下文
            for j in range(max(0, i-2), min(len(sentence), i+3)):
                if j != i:
                    context_words.append(sentence[j])
            
            if context_words:
                training_data.append((target_word, context_words))
    
    # 批量训练
    for epoch in range(100):
        np.random.shuffle(training_data)
        total_loss = 0
        
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i:i+batch_size]
            
            # 获取当前学习率
            lr = scheduler.step(epoch, 100)
            
            # 处理批次
            for target_word, context_words in batch:
                target_idx = vocab[target_word]
                context_indices = [vocab[word] for word in context_words if word in vocab]
                
                if context_indices:
                    model.backward(target_idx, context_indices, learning_rate=lr)
                    
                    # 计算损失
                    output, _ = model.forward(target_idx)
                    loss = -np.sum([np.log(1 / (1 + np.exp(-output[idx])) + 1e-8) 
                                  for idx in context_indices])
                    total_loss += loss
        
        if epoch % 10 == 0:
            avg_loss = total_loss / len(training_data)
            print(f"Epoch {epoch}, LR: {lr:.6f}, Avg Loss: {avg_loss:.4f}")
    
    return model
```

## 模型评估与可视化

### 1. 词向量质量评估

```python
class WordVectorEvaluator:
    def __init__(self, model):
        self.model = model
    
    def evaluate_similarity(self, similarity_dataset):
        """评估词相似度任务"""
        predictions = []
        actual_scores = []
        
        for word1, word2, actual_score in similarity_dataset:
            pred_score = word_similarity(self.model, word1, word2)
            if pred_score is not None:
                predictions.append(pred_score)
                actual_scores.append(actual_score)
        
        # 计算Spearman相关系数
        correlation = np.corrcoef(predictions, actual_scores)[0, 1]
        return correlation
    
    def evaluate_analogy(self, analogy_dataset):
        """评估词类比任务"""
        correct = 0
        total = 0
        
        for analogy in analogy_dataset:
            pos_words, neg_words, expected_word = analogy
            
            result = word_analogy(self.model, pos_words, neg_words, top_n=1)
            if result and result[0][0] == expected_word:
                correct += 1
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        return accuracy
```

### 2. 词向量可视化

```python
def visualize_word_vectors(model, words, method='pca'):
    """词向量可视化"""
    if method == 'pca':
        # PCA降维
        from sklearn.decomposition import PCA
        
        word_vectors = np.array([model.wv[word] for word in words])
        pca = PCA(n_components=2)
        vectors_2d = pca.fit_transform(word_vectors)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
        
        for i, word in enumerate(words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        fontsize=12, alpha=0.8)
        
        plt.title('Word2Vec词向量可视化 (PCA降维)')
        plt.xlabel('第一主成分')
        plt.ylabel('第二主成分')
        plt.grid(True, alpha=0.3)
        plt.show()
        
    elif method == 'tsne':
        # t-SNE降维
        from sklearn.manifold import TSNE
        
        word_vectors = np.array([model.wv[word] for word in words])
        tsne = TSNE(n_components=2, random_state=42)
        vectors_2d = tsne.fit_transform(word_vectors)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], alpha=0.7)
        
        for i, word in enumerate(words):
            plt.annotate(word, (vectors_2d[i, 0], vectors_2d[i, 1]), 
                        fontsize=12, alpha=0.8)
        
        plt.title('Word2Vec词向量可视化 (t-SNE降维)')
        plt.xlabel('t-SNE维度1')
        plt.ylabel('t-SNE维度2')
        plt.grid(True, alpha=0.3)
        plt.show()
```

## 实际应用场景

### 1. 文本相似度计算

```python
def document_similarity(doc1, doc2, model):
    """计算两个文档的相似度"""
    # 分词
    words1 = list(jieba.cut(doc1))
    words2 = list(jieba.cut(doc2))
    
    # 获取词向量
    vecs1 = []
    vecs2 = []
    
    for word in words1:
        if word in model.wv.key_to_index:
            vecs1.append(model.wv[word])
    
    for word in words2:
        if word in model.wv.key_to_index:
            vecs2.append(model.wv[word])
    
    if not vecs1 or not vecs2:
        return 0
    
    # 计算文档向量（平均词向量）
    doc_vec1 = np.mean(vecs1, axis=0)
    doc_vec2 = np.mean(vecs2, axis=0)
    
    # 计算余弦相似度
    return cosine_similarity(doc_vec1, doc_vec2)
```

### 2. 推荐系统应用

```python
class ContentBasedRecommender:
    def __init__(self, model, items):
        self.model = model
        self.items = items
        self.item_vectors = self._compute_item_vectors()
    
    def _compute_item_vectors(self):
        """计算物品的词向量表示"""
        item_vectors = {}
        
        for item_id, description in self.items.items():
            words = list(jieba.cut(description))
            word_vecs = []
            
            for word in words:
                if word in self.model.wv.key_to_index:
                    word_vecs.append(self.model.wv[word])
            
            if word_vecs:
                item_vectors[item_id] = np.mean(word_vecs, axis=0)
        
        return item_vectors
    
    def recommend(self, user_profile, top_n=5):
        """基于用户画像推荐物品"""
        if not user_profile:
            return []
        
        # 计算用户向量
        user_words = list(jieba.cut(user_profile))
        user_vecs = []
        
        for word in user_words:
            if word in self.model.wv.key_to_index:
                user_vecs.append(self.model.wv[word])
        
        if not user_vecs:
            return []
        
        user_vector = np.mean(user_vecs, axis=0)
        
        # 计算与所有物品的相似度
        similarities = []
        for item_id, item_vector in self.item_vectors.items():
            similarity = cosine_similarity(user_vector, item_vector)
            similarities.append((item_id, similarity))
        
        # 排序并返回top_n
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_n]
```

## 总结与展望

Word2Vec通过巧妙的设计，成功地将词汇转换为高维向量表示，实现了语义的数值化。其核心优势包括：

1. **语义保持**：相似的词汇在向量空间中距离相近
2. **数学运算**：支持向量加减运算，实现类比推理
3. **高效训练**：通过负采样等技术大幅提升训练效率
4. **广泛应用**：为后续的NLP模型奠定了坚实基础

### 未来发展方向

1. **上下文感知**：结合Transformer等模型，考虑词汇的上下文信息
2. **多语言支持**：扩展到跨语言的词向量表示
3. **领域适应**：针对特定领域优化词向量质量
4. **动态更新**：支持在线学习和词向量更新

Word2Vec的成功不仅在于其算法创新，更在于其工程实现。它展示了如何将复杂的神经网络模型转化为实用的工具，为自然语言处理的发展开辟了新的道路。

通过深入理解Word2Vec的底层原理，我们不仅能更好地使用这个模型，还能从中学习到宝贵的算法思想和工程经验，为开发自己的词向量模型奠定基础。
