> [Styles of Truncated Backpropagation](https://r2rt.com/styles-of-truncated-backpropagation.html)

> [零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458)

# Styles of Truncated Backpropagation
笔者在之前的博文[Recurrent Neural Networks in Tensorflow](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)中观察到，TensorFlow的截断反向（输入长度为n的阶段子序列）传播方式与“误差最多反向传播n steps”不同。笔者将在这篇博文中基于TensorFlow实现不同的截断反向传播算法，并研究哪种方法更好。

结论是：
- 一个实现得很好、均匀分布的截断反向传播算法与全反向传播（full backpropagation）算法在运行速度上相近，而全反向传播算法在性能上稍微好一点。
- 初步实验表明，n-step Tensorflow-style 反向传播（with num_step=n）并不能有效地反向传播整个n steps误差。因此，如果使用Tensorflow-style 截断反向传播并且需要捕捉n-step依赖关系，则可以使用明显高于n的num_step
以便于有效地将误差反向传播所需地n steps。



