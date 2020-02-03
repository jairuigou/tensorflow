import tensorflow as tf
import numpy as np
from tensorflow import keras

# It has 1 layer, and that layer has 1 neuron, and the input shape to it is just 1 value.
model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
# stochastic gradient descent(sgd) 损失函数用来计算预测结果和正确结果之间的距离
# 优化函数负责减少损失
model.compile(optimizer='sgd',loss='mean_squared_error')

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

model.fit(xs,ys,epochs=5000)

print(10.0,end="-")
print(model.predict([10.0]))