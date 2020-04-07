import tensorflow as tf
import numpy as np
from helpfun.datamanager import DataManager
import matplotlib.pyplot as plt


def build_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1,input_shape=(1,))
    ])
    model.compile(
        optimizer=tf.optimizers.Adam(),
        loss= tf.keras.losses.MeanSquaredError(),
        metrics = ['accuracy']
    )
    return model
if __name__ == '__main__':
    lineardata = DataManager('../data/ex1data1.txt')
    X,y = lineardata.load_data()
    m = len(y)

    model = build_model()

    model.fit(X,y,epochs=3000)

    w,b = model.layers[0].get_weights()
    print(w)
    print(b)

    maxx = np.max(X)
    minx = np.min(X)

    print(maxx)
    print(minx)
    minpredict = model.predict((minx,))
    maxpredict = model.predict((maxx,)) 

    plt.figure("data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.scatter(X,y,c='r',s=10,marker="x")
    plt.plot([minx,maxx],[minpredict[0][0],maxpredict[0][0]])
    plt.savefig("dataplot.png")

     

    
    
