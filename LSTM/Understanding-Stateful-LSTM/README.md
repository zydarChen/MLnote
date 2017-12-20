> [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)
# 任务描述：学习字母表
## 【1】Naive LSTM for Learning One-Char to One-Char Mapping
- 使用t时刻序列值，预测t+1时刻序列值
- 输入维度1，输入用one hot编码
- 效果不好，原因：
  - LSTM没有可以考虑的上下文
  - 每批训练，Keras默认重置网络状态
- 本质上是将LSTM单元用作多层感知机，是对LSTM的误用

## 【2】Naive LSTM for a Three-Char Feature Window to One-Char Mapping
- 使用t-n, t-n+1, ..., t时刻序列值为特征，预测t+1时刻序列值
- 性能小幅提升，但同样不好
- 本质上仍是将LSTM单元用作多层感知机，只不过通过window method来提供上下文，是对LSTM的误用
- 事实上，序列特征是具有time steps的一个特征，而不是one time step的多个特征

## 【3】Naive LSTM for a Three-Char Time Step Window to One-Char Mapping
- 使用t-n, t-n+1, ..., t时刻序列值为time steps特征，预测t+1时刻序列值
- 正确率100%，可以很好的学习到字母表
- 但只能通过前n个序列值预测第n+1个序列值，而不是完整的学习了整张字母表
- 事实上，通过一个足够大的多层感知机也似乎能完成

## 【4】LSTM State Within A Batch
- 使用整个序列训练模型，输入t时刻序列值，预测t+1时刻序列值
- LSTM是有状态的，但在每批训练之后，Keras会默认重置。这意味着，如果我们使用一个足够大的batch，一次训练所有数据，则LSTM会很好的考虑上下文信息
- 正确率100%，可以输入任意字母预测下一个字母
- 问题在于，每批训练都要给网络喂全量数据

## 【5】Stateful LSTM for a One-Char to One-Char Mapping
- 使用整个序列训练模型，输入t时刻序列值，预测t+1时刻序列值
- Ideally, we want to expose the network to the entire sequence and let it learn the inter-dependencies, rather than us define those dependencies explicitly in the framing of the problem.
- 上述通过stateful LSTM来实现，**这才是LSTM的正确用法**
- 手动训练每一期（epoch），每期分为多批，每批训练状态会被记录并传递到下一批，直到本期训练完成后重置状态
- 正确率理论100%（增加epoch可提高正确率）
- 冷启动问题，网络的第一个输入不能被正确预测
- 类似于【3】，但不是人为确定time step，而是由网络自主学习

## 【6】LSTM with Variable-Length Input to One-Char Output
- 使用t时刻前可变序列值，预测t+1时刻序列值
- 定义输入最大time steps，不够最大长度的在前面补0
- 效果一般，但已经能通过任意长度的序列预测下一刻序列值
- 本质上就是【3】，输入改为可变长度