#coding = utf-8
# Recurrent Neural Network 循环神经网络
'''
多用来处理文字和语音，本质上是bp，将上一次的输出结果作为这一次的输入，将语音或文字等序列化数据联系起来
梯度消失问题：随着传播的增加，会忘记之前的输出
LSTM是RNN的一种特殊类型可以学习长期依赖信息
long short term memory
基本结构；输入门，输出门，忘记门
输入门:将输入信号和输入门的数据相乘，得到的结果传递到忘记门
忘记门:决定信号衰减度
输出门:同上
这些门都是经过训练得到的
'''

#以mnist做样例的lstm实现
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

n_inputs = 28 #一行28个数据,输入层有28个神经元
max_time = 28 #一共28行
lstm_size = 100 #隐层单元，隐藏层有100个神经元
n_classes = 10 #10个分类
batch_size = 50 #每个批次50个样本
n_batch = mnist.train.num_examples // batch_size #批次数

x = tf.placeholder(tf.float32,[None,784])
y = tf.placeholder(tf.float32,[None,10])

weights = tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev = 0.1))
biases = tf.Variable(tf.constant(0.1,shape = [n_classes]))

def RNN(X,weights,biases):
    inputs = tf.reshape(X,[-1,max_time,n_inputs])
    #定义lstm基本cell
    lstm_cell = tf.contrib.rnn.core_rnn_cell.BasicLSTMCell(lstm_size)
    #final_state[0]是cell_state
    #final_state[1]是hidden_state
    outputs,final_state = tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1],weights) + biases)
    return results

#计算RNN的返回结果
prediction = RNN(x,weights,biases)
#损失函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#使用AdamOptimizer优化
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一共bool型列表中
correct_prediction = tf.equal(tf.argmax(y-1),tf.argmax(prediction,1))
#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction.tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict = {x:batch_xs,y:batch_ys})

        acc = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels})
        print("Iter" + str(epoch) + ", Testing Accuracy= " + str(acc))


