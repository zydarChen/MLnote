# LSTM
- 经典LSTM网络讲解博文[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，[中文版翻译](http://www.jianshu.com/p/9dc9f41f0b29)
- [Understanding Stateful LSTM Recurrent Neural Networks in Python with Keras](https://machinelearningmastery.com/understanding-stateful-lstm-recurrent-neural-networks-python-keras/)

- rnn_regression.py
  - 莫烦Python
  - 用sin序列拟合cos序列
  - timestep=10，一次迭代中，喂给RNN10个[x, y]，并将上批训练的final_state作为下批训练的init_state，同时在本次迭代预测[x, pre]

- Keras stateful RNN
  - [FAQ: How can I use stateful RNNs?](https://keras.io/getting-started/faq/#how-can-i-use-stateful-rnns)
    - 每次批处理的final_state作为下次批处理的init_state被重用
    - 所有batches的样本量相同
    - 如果`x1`，`x2`是连续的两个batches，那么
    - If x1 and x2 are successive batches of samples, then x2[i] is the follow-up sequence to x1[i], for every i
    - 设置`batch_size`，`stateful=True`，`shuffle=False(fit)`