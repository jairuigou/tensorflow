#coding = utf-8
'''
关于优化器

tf.train.GradientDescentOptimizer(梯度下降法)
标准梯度下降法-计算所有样本汇总误差，根据总误差更新权值
随机梯度下降法(SGD)-随机抽取一个样本来计算误差，更新权值
批量梯度下降法-从所有样本中选取一个批次，根据这个批次的误差来更新权值

Momentum和NAG
tf.train.MomentumOptimizer()

Adagrad

'''