#softmax把输出的结果转换成概率，就是把输出的向量的每一个值n求e的n次方，得到的结果占所有向量的e的n次方的和的比重就算是这个向量值的概率
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt#用来画图

#生成从-0.5到0.5的200个随机点，[]是用来增加这个变量的维度
x_data = np.linspace(-0.5,0.5,200)[:,np.newaxis]
noise = np.random.normal(0,0.02,x_data.shape)
y_data = np.square(x_data) + noise

#定义两个placeholder
x = tf.placeholder(tf.float32,[None,1])
y = tf.placeholder(tf.float32,[None,1])

#构建神经网络
#定义神经网络的中间层
Weights_L1 = tf.Variable(tf.random_normal([1,10]))
biases_L1 = tf.Variable(tf.zeros([1,10]))
Wx_plus_p_L1 = tf.matmul(x,Weights_L1) + biases_L1
L1 = tf.nn.tanh(Wx_plus_p_L1)#双曲正切函数作为激活函数

#定义神经网路的输出层
Weights_L2 = tf.Variable(tf.random_normal([10,1]))
biases_L2 = tf.Variable(tf.zeros([1,1]))
Wx_plus_b_L2 = tf.matmul(L1,Weights_L2) + biases_L2
prediction = tf.nn.tanh(Wx_plus_b_L2)#激活函数

#二次代价函数
loss = tf.reduce_mean(tf.square(y-prediction))
#梯度下降法来训练
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    #变量初始化
    sess.run(tf.global_variables_initializer())
    for _ in range(2000):
        sess.run(train_step,feed_dict = {x:x_data,y:y_data})

    #获得预测值
    prediction_value = sess.run(prediction,feed_dict = {x:x_data})
    #画图
    plt.figure()
    plt.scatter(x_data,y_data)
    plt.plot(x_data,prediction_value,'r-',lw = 5)
    plt.show()
