import tensorflow as tf
import numpy as np

#tensorflow使用示例
#使得构造的线性模型符合给定的线性模型

#使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1 + 0.2

#构造一个线性模型
b = tf.Variable(0.0)
k = tf.Variable(0.0)
y = k*x_data + b

#二次代价函数
#y_data真实值，y样本值，真实值减掉样本值是误差值
#误差的平方求平均值   
loss = tf.reduce_mean(tf.square(y_data-y))
#梯度下降法优化器，学习率0.2
optimizer = tf.train.GradientDescentOptimizer(0.2)
#最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20 == 0:
            print(step,sess.run([k,b]))