import tensorflow as tf

#设置epochs回调函数，当精度达到一定程度时停止训练
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if(logs.get('accuracy') > 0.9):
            print("\ncancell training")
            self.model.stop_training = True

callbacks = myCallback()
mnist = tf.keras.datasets.fashion_mnist
# load_data file path /home/user/.keras/datasets/fashion-mnist
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()

#normalize
training_images = training_images / 255.0
test_images = test_images / 255.0

#define model
# flatten:take the square and turn it into a 1 dimensional set
# dense: add a layer of neurons
# activation 设置激活函数
# relu "if x>0 return x,else return 0"
# softmax picks the biggest one
# Flatten()相当于input层
# softmax层相当于output层，其感知机数量必须和分类种类数量一致
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(600,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])

# compile model with an optimizer and loss function
# metrics 评估标准
model.compile(optimizer = 
    tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
# train
model.fit(training_images,training_labels,epochs=10,callbacks=[callbacks])

model.evaluate(test_images,test_labels)
