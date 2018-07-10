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

#创建神经网络
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#使用softmax函数将结果转换为概率值
prediction = tf.nn.softmax(tf.matmul(x,W)+b)

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#使用梯度下降法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化变量
init = tf.global_variables_initializer()

#确定准确率,结果存放在一个bool型列表中,其中argmax返回一维向量中最大的值所在的位置
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(prediction))
#将上述bool型转换成浮点型，通过reduce_mean求平均值
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(21):
        for batch in range(n_batch):
            #获得batch_size张图片，数据保存在batch_xs中，标签保存在batchys中
            batch_xs,batchys = mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batchys})
        
        acc = sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy " + str(acc))