#coding = utf-8
'''
卷积神经网络的基本操作

1->卷积层
2->池化层
max-pooling 最大池化
mean-pooling 随机池化

卷积核采样之后得到的特征图比原图小，需要补成原图大小
same padding----可能会给平面外部补零
valid padding----不会超出平面外部

一般会先有一个卷积层，之后池化层，再之后卷积层池化层.....最后会有一个全连接层
'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

#每个批次的大小
batch_size = 100
#计算总共有多少个批次
n_batch = mnist.train.num_examples // batch_size

#初始化权值
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)#生成一个截断的正态分布
    return tf.Variable(initial)

#初始化b值
def bias_variable(shape):
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#卷积层
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

#池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,784])
x = tf.placeholder(tf.float32,[None,10])

#改变x的格式转为4D向量
x_image = tf.reshape(x,[-1,28,28,1])

#初始化第一个卷积层的权值和偏置
W_conv1 = weight_variable([5,5,1,32])#5*5的采样窗口，32个卷积核从一个平面抽取特征
b_conv1 = bias_variable([32])#每个卷积核一个偏置值

#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

#初始化第二个卷积层的权值和偏置
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

#重复操作
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#初始化第一个全连接层的权值
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

#把池化层2的输出扁平化为1维
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
#计算第一个全连接层的输出
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)

#keep_prob用来表示神经元的输出概率
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#初始化第二个全连接层
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])

#计算输出
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction),tf.float32)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batsh_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7})

        ass = sess.run(accuracy,feed_dict = {x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("Iter" + str(epoch) + ", Testing Accuracy= " + str(acc))

