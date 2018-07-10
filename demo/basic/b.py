import tensorflow as tf

x = tf.Variable([1,2])
a = tf.constant([3,3])

sub = tf.subtract(x,a)
add = tf.add(x,sub)

#全局变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))

#创建一个变量初始化为0
state = tf.Variable(0,name='counter')
new_value = tf.add(state,1)
#赋值方法
update = tf.assign(state,new_value)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(6):
        #run方法，单独使用run(state)会执行赋值和加一两个操作，如果在之前单独run(new_value)效果一样
        sess.run(new_value)
        sess.run(update)
        print(sess.run(state))
    print(sess.run(state))