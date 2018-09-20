Bilateral Multi-Perspective Matching for Natural Language Sentences*

这篇论文中提出双向多角度匹配模型(***BiMPM***),主要步骤包括使用***BiLSTM***编码两个句子，在双向对已编码的句子进行匹配，在每个匹配方向上，在多个角度上句子每个时间步都与另一个句子在所有时间步上匹配，另一个***BiLSTM***层用于聚合匹配结果为一个固定长度的匹配向量，最后使用全连接进行最后的处理。本文在三个任务上进行验证：歧义识别，自然语言推理，答案句子选择，实验结果表明该模型在所有获得最先进的性能。

#### 任务说明

​      描述每个自然语言句子匹配任务为三元组,\(P,Q,y\),其中$P=(p_1, p_2, ..., p_m)$,$Q=(q_1,q_2,...,q_N)$,$y\in \mathcal{Y}$是P和Q的标签描述,目标是:

​                $$y{*}=arg max_{y\in \mathcal{Y}}Pr(y|P,Q)$$

#### 模型架构

![](F:\远传\知识图谱\图片1.png)

​    提出的BiMPM模型是为了估计$Pr(y|P,Q)$的概率分布。整体包括词描述层，上下文描述层，匹配层，融合层，预测层。

##### 词描述层

​    本文对句子编码通过词嵌入和字符嵌入相结合的方式。词嵌入通过word2vec或者GloVe，字符嵌入通过在线训练的方式获取，初始时随机初始化。

##### 上下文描述层

​    这层目的是整合上下文信息到P和Q的时间步中，本文使用BiLSTM对句子P每时间步进行编码：

​                          $$\vec{h_i^p}=\vec{LSTM}(\vec{h_{i-1}^p}, p_i)$$

​                          $$\overleftarrow{h_i^p} \quad = \overleftarrow{LSTM}(\overleftarrow{h_{i+1}^p}\quad,p_i)\quad$$

##### 匹配层

该层的目的是比较一个句子的每个时间步与另一个句子所有上下文嵌入（时间步）。我们匹配两个句子P和Q在两个方向：P的每个时间步与Q所有时间步匹配，Q的每个时间步与P所有时间步匹配。多角度匹配层的输出是两个句子的匹配向量，每个匹配向量是一个句子时间步与另一个句子所有时间步的匹配结果。

##### 融合层

这层用于融合两个匹配向量为一个固定长度的匹配向量。本文使用另一个BiLSTM分别用于两个匹配向量的序列。重构固定长度匹配向量通过连接最后时间步向量。

##### 预测层

这层的目标是求值概率分布$P_r(y|P,Q)$.

#### 多角度匹配操作

   这个操作包括两步：

   第一步，本文定义一个多角度余弦匹配函数$f_m$来匹配两个向量：

 $$m=f_m(v_i,v_2;W)$$

其中$v_1$,$v_2$是两个d维度向量，$W \in \mathcal{R^{l \times d}}$.m是一个l维向量$m=[m_1,...,m_k,...,m_l]$.$m_k$是通过累积两个权重向量的余弦相似度。

 $$m_k=cosine(W_k \circ v_1, W_k \circ_2)$$

第二步，基于$f_m$，本文定义四个匹配策略来比较一个句子每个时间步与另一个句子所有时间步。

（1）全匹配。图2(a)显示这个匹配策略。在这种策略中，每个前向（或者后向）上下文嵌入$\vec{h_i^p}$(或者$\overleftarrow{h_i^q}\quad$）来与另一个句子的前向（或者后向）最后时间步$\vec{h_N^q}$（或者$\overleftarrow{h_1^q}$)进行比较:

$$\vec{m_i}^{full}=f_m(\vec{h_i}^p,\vec{h_N}^q;W_1)$$

 $$\overleftarrow{m_i}^{full}\quad=f_m(\overleftarrow{h_i}  ^p\quad,\overleftarrow{h_1}^q\quad;W_2)$$

(2)最大池化匹配。图2（b）显示这个匹配策略。这个策略中，每个前向（或者后向）上下文嵌入$h_i^p$(或者$\overleftarrow{h_i^p}$）与另一个句子每个前向（或者后向）上下文嵌入$\vec{h_j^p}$(或者$\overleftarrow{h_j^q}\quad$)进行比较.只保留每一维的最大值。![图片2](F:\远传\知识图谱\图片2.png)

图2 不同匹配策略

$$\vec{m_i}=max_{j \in \mathcal(1,...,N)}f_m(\vec{h_i^p},\vec{h_j^1};W^3)$$

$$\overleftarrow{m_i^{max}}\quad=max_{j \in \mathcal (1, ..., N)}f_m(\overleftarrow{h_i^p}\quad,\overleftarrow{h_j^q};W^4)$$

(3)注意力匹配。图2（c）显示这种匹配策略。本文
$$
\overleftarrow{m_i^{max}}\quad=max_{j \in \mathcal (1, ..., N)}f_m(\overleftarrow{h_i^p}\quad,\overleftarrow{h_j^q};W^4)
$$





















