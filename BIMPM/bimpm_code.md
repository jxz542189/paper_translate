### Bilateral Multi-Perspective Matching for Natural Language Sentences代码解析

https://github.com/zhiguowang/BiMPM

这个模型接口为

```python
def create_model_graph(self, num_classes, word_vocab=None, char_vocab=None, is_training=True, global_step=None):
    '''
    num_classes:分类类别数
    word_vocab:字典，包括每个字对应的字嵌入，可以是已训练的字向量也可以是在线训练的字向量
    char_vocab:字符典，
    is_training:是否训练
    global_step:记录全局训练步，可以通过tf.train.get_or_create_global_step()获取
    '''
```

如果字典不为空的话，就会将对问题和段落按照字进行字嵌入

```python
if word_vocab is not None:
    word_vec_trainable = True
    cur_device = '/gpu:0'
    if options.fix_word_vec:
        word_vec_trainable = False
        cur_device = '/cpu:0'
        with tf.device(cur_device):
            self.word_embedding = tf.get_variable("word_embedding",
                                                  trainable=word_vec_trainable,
                                                  initializer=tf.constant(word_vocab.word_vecs),
                                                  dtype=tf.float32)

            in_question_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_question_words) # [batch_size, question_len, word_dim]
            in_passage_word_repres = tf.nn.embedding_lookup(self.word_embedding, self.in_passage_words) # [batch_size, passage_len, word_dim]
            in_question_repres.append(in_question_word_repres)
            in_passage_repres.append(in_passage_word_repres)

            input_shape = tf.shape(self.in_question_words)
            batch_size = input_shape[0]
            question_len = input_shape[1]
            input_shape = tf.shape(self.in_passage_words)
            passage_len = input_shape[1]
            input_dim += word_vocab.word_dim
```

如果使用字符嵌入和字符字典不为空的话，这里会对问题和段落分别进行字符嵌入，然后进行mask掩饰，再分别通过bilstm进行编码，对对前向的最后一个时间步的输出和后向第一时间步的输出进行整合在一起。

```python
if options.with_char and char_vocab is not None:
    input_shape = tf.shape(self.in_question_chars)
    batch_size = input_shape[0]
    question_len = input_shape[1]
    q_char_len = input_shape[2]
    input_shape = tf.shape(self.in_passage_chars)
    passage_len = input_shape[1]
    p_char_len = input_shape[2]
    char_dim = char_vocab.word_dim
    self.char_embedding = tf.get_variable("char_embedding", initializer=tf.constant(char_vocab.word_vecs), dtype=tf.float32)

    in_question_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_question_chars) # [batch_size, question_len, q_char_len, char_dim]
    in_question_char_repres = tf.reshape(in_question_char_repres, shape=[-1, q_char_len, char_dim])
    question_char_lengths = tf.reshape(self.question_char_lengths, [-1])
    quesiton_char_mask = tf.sequence_mask(question_char_lengths, q_char_len, dtype=tf.float32)  # [batch_size*question_len, q_char_len]
    in_question_char_repres = tf.multiply(in_question_char_repres, tf.expand_dims(quesiton_char_mask, axis=-1))

    in_passage_char_repres = tf.nn.embedding_lookup(self.char_embedding, self.in_passage_chars) # [batch_size, passage_len, p_char_len, char_dim]
    in_passage_char_repres = tf.reshape(in_passage_char_repres, shape=[-1, p_char_len, char_dim])
    passage_char_lengths = tf.reshape(self.passage_char_lengths, [-1])
    passage_char_mask = tf.sequence_mask(passage_char_lengths, p_char_len, dtype=tf.float32)  # [batch_size*passage_len, p_char_len]
    in_passage_char_repres = tf.multiply(in_passage_char_repres, tf.expand_dims(passage_char_mask, axis=-1))

    (question_char_outputs_fw, question_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_question_char_repres, options.char_lstm_dim,
                                                                                        input_lengths=question_char_lengths,scope_name="char_lstm", reuse=False,
                                                                                        is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
    question_char_outputs_fw = layer_utils.collect_final_step_of_lstm(question_char_outputs_fw, question_char_lengths - 1)
    question_char_outputs_bw = question_char_outputs_bw[:, 0, :]
    question_char_outputs = tf.concat(axis=1, values=[question_char_outputs_fw, question_char_outputs_bw])
    question_char_outputs = tf.reshape(question_char_outputs, [batch_size, question_len, 2*options.char_lstm_dim])

    (passage_char_outputs_fw, passage_char_outputs_bw, _) = layer_utils.my_lstm_layer(in_passage_char_repres, options.char_lstm_dim,
                                                                                      input_lengths=passage_char_lengths,scope_name="char_lstm", reuse=True,                                       is_training=is_training, dropout_rate=options.dropout_rate, use_cudnn=options.use_cudnn)
    passage_char_outputs_fw = layer_utils.collect_final_step_of_lstm(passage_char_outputs_fw, passage_char_lengths - 1)
    passage_char_outputs_bw = passage_char_outputs_bw[:, 0, :]
    passage_char_outputs = tf.concat(axis=1, values=[passage_char_outputs_fw, passage_char_outputs_bw])
    passage_char_outputs = tf.reshape(passage_char_outputs, [batch_size, passage_len, 2*options.char_lstm_dim])

    in_question_repres.append(question_char_outputs)
    in_passage_repres.append(passage_char_outputs)

    input_dim += 2 * options.char_lstm_dim
```

随后根据是否使用highway，这里主要是为了更好训练层数比较深的神经网络，问题和段落共享相同的网络参数：



```python
        if options.with_highway:
            with tf.variable_scope("input_highway"):
                in_question_repres = match_utils.multi_highway_layer(in_question_repres, input_dim, options.highway_layer_num)
                tf.get_variable_scope().reuse_variables()
                in_passage_repres = match_utils.multi_highway_layer(in_passage_repres, input_dim, options.highway_layer_num)

```

接下来才是本文核心代码，双向匹配函数:

```python
def bilateral_match_func(in_question_repres,
                         in_passage_repres,
                         question_lengths,
                         passage_lengths, 
                         question_mask, 
                         passage_mask, 
                         input_dim, 
                         is_training, 
                         options=None):
    '''
    in_question_repres:[batch_size, question_len, dim]
    in_passage_repres:[batch_size, passage_len, dim]
    question_lengths:[batch_size]
    passage_lengths:[batch_size]
    question_mask:[batch_size,question_len]
    passage_mask:[batch_size,passage_len]
    input_dim:dim
    '''
```

在双向匹配函数中