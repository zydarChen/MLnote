# LSTM
- 经典LSTM网络讲解博文[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)，[中文版翻译](http://www.jianshu.com/p/9dc9f41f0b29)

- rnn_regression.py
 - 莫烦Python
 - 用sin序列拟合cos序列
 - timestep=10，一次迭代中，喂给RNN10个[x, y]，并将上批训练的final_state作为下批训练的init_state，同时在本次迭代预测[x, pre]