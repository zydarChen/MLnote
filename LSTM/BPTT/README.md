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
![RNN_true_truncated_backprop.png](../res/RNN_true_truncated_backprop.png)
下图是Tensorflow-style截断反向传播在同一序列上的情况：
![RNN_tf_truncated_backprop.png](../res/RNN_tf_truncated_backprop.png)



