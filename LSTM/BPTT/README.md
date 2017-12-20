> [Styles of Truncated Backpropagation](https://r2rt.com/styles-of-truncated-backpropagation.html)
> [零基础入门深度学习(5) - 循环神经网络](https://zybuluo.com/hanbingtao/note/541458)

# Styles of Truncated Backpropagation
笔者在之前的博文[Recurrent Neural Networks in Tensorflow](https://r2rt.com/recurrent-neural-networks-in-tensorflow-i.html)中观察到，TensorFlow的截断反向（输入长度为n的阶段子序列）传播方式与“误差最多反向传播n steps”不同。笔者将在这篇博文中基于TensorFlow中实现不同的截断反向传播算法，并研究哪种方法更好。
结论是：

