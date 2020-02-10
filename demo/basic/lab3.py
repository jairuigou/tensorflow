'''
That's the concept of Convolutional Neural Networks. Add some layers to do convolution before you have the dense layers, 
and then the information going to the dense layers becomes more focused and possibly more accurate.
'''
#使用cnn提高lab2的准确度
import tensorflow as tf
mnist = tf.keras.datasets.fashion_mnist
(training_images,training_labels),(test_images,test_labels) = mnist.load_data()
#normalize
training_images=training_images.reshape(60000,28,28,1)
training_images = training_images / 255.0
test_images = test_images.reshape(10000,28,28,1)
test_images = test_images / 255.0
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128,activation=tf.nn.relu),
    tf.keras.layers.Dense(10,activation=tf.nn.softmax)
])
model.compile(optimizer = 
    tf.optimizers.Adam(),
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)
# train
model.fit(training_images,training_labels,epochs=5)

test_loss,test_accuracy = model.evaluate(test_images,test_labels)
print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))