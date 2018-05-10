# LSTM
## 学习

- Udacity[深度学习课程](https://classroom.udacity.com/courses/ud730/lessons/6378983156/concepts/63742734590923)
  - [笔记](./Udacity)
- 大白话RNN


- 经典LSTM网络讲解博文[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，[中文版翻译](http://www.jianshu.com/p/9dc9f41f0b29)
- [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)
  - [笔记](./Understanding-Stateful-LSTM)
- Keras stateful RNN
  - [FAQ: How can I use stateful RNNs?](https://keras.io/getting-started/faq/#how-can-i-use-stateful-rnns)
    - 每次批处理的final_state作为下次批处理的init_state被重用
    - 所有batches的样本量相同
    - 如果`x1`，`x2`是连续的两个batches，那么
    - If x1 and x2 are successive batches of samples, then x2[i] is the follow-up sequence to x1[i], for every i
    - 设置`batch_size`，`stateful=True`，`shuffle=False(fit)`
- [截断反向传播](./BPTT)

## Demo

- [Morvan_rnn_regression.py](https://morvanzhou.github.io/tutorials/machine-learning/tensorflow/5-09-RNN3/)
  - 莫烦Python
  - 用sin序列拟合cos序列
  - 50batch，20step，每次训练1000时间点
  - x_batch shape (50,20,1) Y_batch shape (50, 20, 1)
  - 第一个batch[0, 999]，第二个batch[20, 1019]
  - timestep=10，一次迭代中，喂给RNN10个[x, y]，并将上批训练的final_state作为下批训练的init_state，同时在本次迭代预测[x, pre]
- blog:  [Multivariate Time Series Forecasting with LSTMs in Keras](https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/)
- blog: [LSTM Neural Network for Time Series Prediction](http://www.jakob-aungiers.com/articles/a/LSTM-Neural-Network-for-Time-Series-Prediction)