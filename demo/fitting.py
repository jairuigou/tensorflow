#coding=utf-8
"""
关于欠拟合(underfitting)，正确拟合(just right),过拟合(overfitting)

一般出现过拟合问题是由于网络模型过于复杂，而数据量太少导致模型不能确定参数
解决过拟合
1.更多的数据集
2.正则化方法
3.Dropout-随机切换工作的神经元，每次迭代不是全部的神经元都起作用,会使收敛速度
"""

#使用mnist训练测试Dropout方法
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#载入数据集
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

#每个批次的大小(一次性向神经网络中放入100张图片)
batch_size = 100
#计算总共的批次数量,//表示整除
n_batch = mnist.train.num_examples // batch_size

#定义两个placeholder,784 = 28*28即一张图片的向量
x = tf.placeholder(tf.float32,[None,784])
#标签，0~9
y = tf.placeholder(tf.float32,[None,10])

keep_prob=tf.placeholder(tf.float32)


#创建神经网络
'''
W1 = tf.Variable(tf.zeros([784,10]))
b1 = tf.Variable(tf.zeros([10]))
更改初始化方式
'''
W1 = tf.Variable(tf.truncated_normal([784,2000],stddev=0.1))
b1 = tf.Variable(tf.zeros([2000])+0.1)
#设置隐藏层，使用双曲正切函数进行激活
L1 = tf.nn.tanh(tf.matmul(x,W1)+b1)
#调用dropout方法,keep_prob参数代表L1层中有百分之多少的神经元是工作的
L1_drop = tf.nn.dropout(L1,keep_prob)

#单层2000个神经元并不需要，只是为了学习用
W2 = tf.Variable(tf.truncated_normal([2000,2000],stddev=0.1))
b2 = tf.Variable(tf.zeros([2000])+0.1)
L2 = tf.nn.tanh(tf.matmul(L1_drop,W2)+b2)
L2_drop = tf.nn.dropout(L2,keep_prob)

W3 = tf.Variable(tf.truncated_normal([2000,1000],stddev=0.1))
b3 = tf.Variable(tf.zeros([1000])+0.1)
L3 = tf.nn.tanh(tf.matmul(L2_drop,W3)+b3)
L3_drop = tf.nn.dropout(L3,keep_prob)

W4 = tf.Variable(tf.truncated_normal([1000,10],stddev=0.1))
b4 = tf.Variable(tf.zeros([10])+0.1)
#使用softmax函数将结果转换为概率值
prediction = tf.nn.softmax(tf.matmul(L3_drop,W4)+b4)

#对数似然代价函数,loss等于交叉熵的平均值
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#确定准确率,结果存放在一个bool型列表中,其中argmax返回一维向量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
#将上述bool型转换成浮点型，通过reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            #获得batch_size张图片，数据保存在batch_xs中，标签保存在batchys中
            batch_xs,batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batchys,keep_prob:1.0})
        
        test_acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:0.7})
        train_acc = sess.run(accuracy,feed_dict={x:mnist.train.images,y:mnist.train.labels,keep_prob:0.7})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(test_acc) + ",Training Accuracy " + str(train_acc))