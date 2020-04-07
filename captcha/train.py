import tensorflow as tf
import numpy as np
from captchaimg import CaptchaImg

if __name__ == '__main__':
    captcha = CaptchaImg()
    training_images,training_labels = captcha.load_train_data()
    test_images,test_labels = captcha.load_test_data()

    input_shape0 = training_images.shape[0]
    input_shape1 = training_images.shape[1]
    input_shape2 = training_images.shape[2]
    test_shape0 = test_images.shape[0]
    test_shape1 = test_images.shape[1]
    test_shape2 = test_images.shape[2]
    training_images = training_images.reshape(input_shape0,input_shape1,input_shape2,1)
    training_images = training_images / 255.0
    test_images = test_images.reshape(test_shape0,test_shape1,test_shape2,1)
    test_images = test_images / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu,input_shape=(input_shape1,input_shape2,1)),
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

    model.fit(training_images,training_labels,epochs=2)

    test_loss,test_accuracy = model.evaluate(test_images,test_labels)
    print ('Test loss: {}, Test accuracy: {}'.format(test_loss, test_accuracy*100))

    model.save('captcha_model')
