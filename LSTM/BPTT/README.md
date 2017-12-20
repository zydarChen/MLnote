> 本文[GitHub地址](https://github.com/zydarChen/MLnote/tree/master/LSTM/BPTT)

> [Styles of Truncated Backpropagation](https://r2rt.com/styles-of-truncated-backpropagation.html)

> [零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458)

# Styles of Truncated Backpropagation
笔者在之前的博文[Recurrent Neural Networks in Tensorflow](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)中观察到，TensorFlow的截断反向（输入长度为n的阶段子序列）传播方式与“误差最多反向传播n steps”不同。笔者将在这篇博文中基于TensorFlow实现不同的截断反向传播算法，并研究哪种方法更好。

结论是：
- 一个实现得很好、均匀分布的截断反向传播算法与全反向传播（full backpropagation）算法在运行速度上相近，而全反向传播算法在性能上稍微好一点。
- 初步实验表明，n-step Tensorflow-style 反向传播（with num_step=n）并不能有效地反向传播整个n steps误差。因此，如果使用Tensorflow-style 截断反向传播并且需要捕捉n-step依赖关系，则可以使用明显高于n的num_step
以便于有效地将误差反向传播所需地n steps。

## Differences in styles of truncated backpropagation
假设要训练一个RNN，训练集为长度10,000的序列。如果我们使用非截断反向传播，那么整个序列会一次性地喂给网络，在10,000时刻的误差会被一直传递到时刻1。这会导致两个问题：
1. 误差反向传播这么多步的计算开销大
2. 由于梯度消失，反向传播误差逐层变小，使得进一步的反向传播不再重要

可以使用“截断”反向传播解决这个问题，Ilya Sutskever博士[这篇](http://202.38.196.91/cache/6/03/www.cs.utoronto.ca/96b2e707823b366dbfe710eeeedf95b3/ilya_sutskever_phd_thesis.pdf)文章的2.8.6节有一个对反向传播很好的描述：
> “[Truncated backpropagation] processes the sequence one timestep at a time, and every k1 timesteps, it runs BPTT for k2 timesteps…”

Tensorflow-style反向传播使用`k1=k2(=num_steps)`（详情请参阅[Tensorflow Api](https://www.tensorflow.org/tutorials/recurrent#truncated-backpropagation)）。本文提出的问题是：`k1=1`能否得到一个更好的结果。本文认为，“真”截断反向传播是：每次反向传播k2 steps都是传播了整个k2 steps。

举个例子解释这两种方法的不同。序列长度为49，反向传播截断为7 steps。相同的是，每个误差都会反向传播7 steps。但是，对于Tensorflow-style截断反向传播，序列会切分成长度为7的7个子序列，并且只有7个误差被反向传播了7 steps。而对于“真”截断反向传播而言，42个误差能被反向传播7 steps就应该42个都反向传播7 steps。这就产生了不同，因为使用7-steps与1-steps更新权重明显不同。

为了可视化差异，下图表示的是序列长度为6，误差反向传播3 steps：

![RNN_true_truncated_backprop.png](../../res/RNN_true_truncated_backprop.png)

下图是Tensorflow-style截断反向传播在同一序列上的情况：

![RNN_tf_truncated_backprop.png](../../res/RNN_tf_truncated_backprop.png)

## 实验设计
为了比较两种算法的性能，笔者实现了一个“真”截断反向传播算法，并与vanilla-RNN比较结果。vanilla-RNN是笔者在先前的博文[Recurrent Neural Networks in Tensorflow I](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)中使用的模型。不同的是，先前的网络在toy数据集上学习简单模式非常快，而本次提升了任务和模型的复杂性。这次是任务对ptb数据集进行语言建模，在基本的RNN模型中添加了一个embedding层和dropout层用以匹配任务的复杂性。

实验比较了每个算法20 epochs训练后验证集上的最佳性能，使用的是Adam Optimizer（初步实验它比其他Optimizer好），学习率设置为0.003，0.001和0.0003。

- 5-step truncated backpropagation
  - True, sequences of length 20
  - TF-style
- 10-step truncated backpropagation
  - True, sequences of length 30
  - TF-style
- 20-step truncated backpropagation
  - True, sequences of length 40
  - TF-style
- 40-step truncated backpropagation
  - TF-style

## 代码
### Imports and data generators
```python
import numpy as np
import tensorflow as tf
%matplotlib inline
import matplotlib.pyplot as plt
from tensorflow.models.rnn.ptb import reader

#data from http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz
raw_data = reader.ptb_raw_data('ptb_data')
train_data, val_data, test_data, num_classes = raw_data

def gen_epochs(n, num_steps, batch_size):
    for i in range(n):
        yield reader.ptb_iterator(train_data, batch_size, num_steps)
```

### Model
```python
def build_graph(num_steps,
            bptt_steps = 4, batch_size = 200, num_classes = num_classes,
            state_size = 4, embed_size = 50, learning_rate = 0.01):
    """
    Builds graph for a simple RNN

    Notable parameters:
    num_steps: sequence length / steps for TF-style truncated backprop
    bptt_steps: number of steps for true truncated backprop
    """

    g = tf.get_default_graph()

    # placeholders
    x = tf.placeholder(tf.int32, [batch_size, None], name='input_placeholder')
    y = tf.placeholder(tf.int32, [batch_size, None], name='labels_placeholder')
    default_init_state = tf.zeros([batch_size, state_size])
    init_state = tf.placeholder_with_default(default_init_state,
                                             [batch_size, state_size], name='state_placeholder')
    dropout = tf.placeholder(tf.float32, [], name='dropout_placeholder')

    x_one_hot = tf.one_hot(x, num_classes)
    x_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, x_one_hot)]

    with tf.variable_scope('embeddings'):
        embeddings = tf.get_variable('embedding_matrix', [num_classes, embed_size])

    def embedding_lookup(one_hot_input):
        with tf.variable_scope('embeddings', reuse=True):
            embeddings = tf.get_variable('embedding_matrix', [num_classes, embed_size])
            embeddings = tf.identity(embeddings)
            g.add_to_collection('embeddings', embeddings)
            return tf.matmul(one_hot_input, embeddings)

    rnn_inputs = [embedding_lookup(i) for i in x_as_list]

    #apply dropout to inputs
    rnn_inputs = [tf.nn.dropout(x, dropout) for x in rnn_inputs]

    # rnn_cells
    with tf.variable_scope('rnn_cell'):
        W = tf.get_variable('W', [embed_size + state_size, state_size])
        b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))

    def rnn_cell(rnn_input, state):
        with tf.variable_scope('rnn_cell', reuse=True):

            W = tf.get_variable('W', [embed_size + state_size, state_size])
            W = tf.identity(W)
            g.add_to_collection('Ws', W)

            b = tf.get_variable('b', [state_size], initializer=tf.constant_initializer(0.0))
            b = tf.identity(b)
            g.add_to_collection('bs', b)

            return tf.tanh(tf.matmul(tf.concat(1, [rnn_input, state]), W) + b)

    state = init_state
    rnn_outputs = []
    for rnn_input in rnn_inputs:
        state = rnn_cell(rnn_input, state)
        rnn_outputs.append(state)

    #apply dropout to outputs
    rnn_outputs = [tf.nn.dropout(x, dropout) for x in rnn_outputs]

    final_state = rnn_outputs[-1]

    #logits and predictions
    with tf.variable_scope('softmax'):
        W = tf.get_variable('W_softmax', [state_size, num_classes])
        b = tf.get_variable('b_softmax', [num_classes], initializer=tf.constant_initializer(0.0))
    logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
    predictions = [tf.nn.softmax(logit) for logit in logits]

    #losses
    y_as_list = [tf.squeeze(i, squeeze_dims=[1]) for i in tf.split(1, num_steps, y)]
    losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logit,label) \
              for logit, label in zip(logits, y_as_list)]
    total_loss = tf.reduce_mean(losses)

    """
    Implementation of true truncated backprop using TF's high-level gradients function.

    Because I add gradient-ops for each error, this are a number of duplicate operations,
    making this a slow implementation. It would be considerably more effort to write an
    efficient implementation, however, so for testing purposes, it's OK that this goes slow.

    An efficient implementation would still require all of the same operations as the full
    backpropagation through time of errors in a sequence, and so any advantage would not come
    from speed, but from having a better distribution of backpropagated errors.
    """

    embed_by_step = g.get_collection('embeddings')
    Ws_by_step = g.get_collection('Ws')
    bs_by_step = g.get_collection('bs')

    # Collect gradients for each step in a list
    embed_grads = []
    W_grads = []
    b_grads = []

    # Keeping track of vanishing gradients for my own curiousity
    vanishing_grad_list = []

    # Loop through the errors, and backpropagate them to the relevant nodes
    for i in range(num_steps):
        start = max(0,i+1-bptt_steps)
        stop = i+1
        grad_list = tf.gradients(losses[i],
                                 embed_by_step[start:stop] +\
                                 Ws_by_step[start:stop] +\
                                 bs_by_step[start:stop])
        embed_grads += grad_list[0 : stop - start]
        W_grads += grad_list[stop - start : 2 * (stop - start)]
        b_grads += grad_list[2 * (stop - start) : ]

        if i >= bptt_steps:
            vanishing_grad_list.append(grad_list[stop - start : 2 * (stop - start)])

    grad_embed = tf.add_n(embed_grads) / (batch_size * bptt_steps)
    grad_W = tf.add_n(W_grads) / (batch_size * bptt_steps)
    grad_b = tf.add_n(b_grads) / (batch_size * bptt_steps)

    """
    Training steps
    """

    opt = tf.train.AdamOptimizer(learning_rate)
    grads_and_vars_tf_style = opt.compute_gradients(total_loss, tf.trainable_variables())
    grads_and_vars_true_bptt = \
        [(grad_embed, tf.trainable_variables()[0]),
         (grad_W, tf.trainable_variables()[1]),
         (grad_b, tf.trainable_variables()[2])] + \
        opt.compute_gradients(total_loss, tf.trainable_variables()[3:])
    train_tf_style = opt.apply_gradients(grads_and_vars_tf_style)
    train_true_bptt = opt.apply_gradients(grads_and_vars_true_bptt)

    return dict(
        train_tf_style = train_tf_style,
        train_true_bptt = train_true_bptt,
        gvs_tf_style = grads_and_vars_tf_style,
        gvs_true_bptt = grads_and_vars_true_bptt,
        gvs_gradient_check = opt.compute_gradients(losses[-1], tf.trainable_variables()),
        loss = total_loss,
        final_state = final_state,
        x=x,
        y=y,
        init_state=init_state,
        dropout=dropout,
        vanishing_grads=vanishing_grad_list
    )
```

## Some quick tests
### 速度测试
不出所料，由于执行了重复操作，实现的真BPTT速度很慢。一个有效的实现运行速度将与全反向传播大致相同。

```python
reset_graph()
g = build_graph(num_steps = 40, bptt_steps = 20)
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

X, Y = next(reader.ptb_iterator(train_data, batch_size=200, num_steps=40))
```

```python
%%timeit
gvs_bptt = sess.run(g['gvs_true_bptt'], feed_dict={g['x']:X, g['y']:Y, g['dropout']: 1})
```

```python
10 loops, best of 3: 173 ms per loop
```

```python
%%timeit
gvs_tf = sess.run(g['gvs_tf_style'], feed_dict={g['x']:X, g['y']:Y, g['dropout']: 1})
```

```python
10 loops, best of 3: 80.2 ms per loop
```

### 梯度消失演示
为了掩饰梯度消失问题，收集以下信息。如你所见，梯度很快就会消失，每一步都会减少3-4倍。

```python
vanishing_grads, gvs = sess.run([g['vanishing_grads'], g['gvs_true_bptt']],
                                feed_dict={g['x']:X, g['y']:Y, g['dropout']: 1})
vanishing_grads = np.array(vanishing_grads)
weights = gvs[1][1]

# sum all the grads from each loss node
vanishing_grads = np.sum(vanishing_grads, axis=0)

# now calculate the l1 norm at each bptt step
vanishing_grads = np.sum(np.sum(np.abs(vanishing_grads),axis=1),axis=1)

vanishing_grads
```
```python
array([  5.28676978e-08,   1.51207473e-07,   4.04591049e-07,
         1.55859300e-06,   5.00411124e-06,   1.32292716e-05,
         3.94736344e-05,   1.17605050e-04,   3.37805774e-04,
         1.01710076e-03,   2.74375151e-03,   8.92040879e-03,
         2.23708227e-02,   7.23497868e-02,   2.45202959e-01,
         7.39126682e-01,   2.19093657e+00,   6.16793633e+00,
         2.27248211e+01,   9.78200531e+01], dtype=float32)
```

```python
for i in range(len(vanishing_grads) - 1):
    print(vanishing_grads[i+1] / vanishing_grads[i])
```

    2.86011
	2.67573
	3.85227
	3.21066
	2.64368
	2.98381
	2.97933
	2.87237
	3.0109
	2.69762
	3.25117
	2.50782
	3.23411
	3.38913
	3.01435
	2.96422
	2.81521
	3.68435
	4.30455