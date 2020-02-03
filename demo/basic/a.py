import tensorflow as tf

#程序结构分为定义变量和会话部分，所有变量在会话部分执行初始化和运算操作 

#常量op,m1一行两列,m2两行一列
m1 = tf.constant([[3,3]])
m2 = tf.constant([[2],[3]])

product = tf.matmul(m1,m2)

#输出结果不是一个数字，而是一个tensor
print(product)

with tf.compat.v1.Session() as sess:
    m1 = tf.constant([[3,3]])
    m2 = tf.constant([[2],[3]])
    c = tf.matmul(m1,m2)
    result = sess.run(c)
    print(result)