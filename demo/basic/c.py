import tensorflow as tf

#Fetch 在会话中同时运行多个op
input1 = tf.constant(3.0)
input2 = tf.constant(3.0)
input3 = tf.constant(3.0)

add = tf.add(input2,input3)
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([mul,add])
    print(result)

#Feed
input1 = tf.placeholder(tf.float32)#占位符
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #Feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7.0],input2:[2.0]}))