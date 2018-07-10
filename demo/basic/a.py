import tensorflow as tf

#程序结构分为定义变量和会话部分，所有变量在会话部分执行初始化和运算操作 

#常量op,m1一行两列,m2两行一列
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

product = tf.matmul(m1,m2)

#输出结果不是一个数字，而是一个tensor
print(product)

#定义一个会话，启动默认图
sess = tf.Session()
result = sess.run(product)
print(result)
#关闭会话
sess.close()

#不需要关闭会话
with tf.Session() as sess:
    result = sess.run(product)
    print(result)