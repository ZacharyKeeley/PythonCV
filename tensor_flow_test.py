'''
import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
'''

import tensorflow as tf
import sys
import os
import matplotlib.pyplot as plt
import numpy as np


if __name__== "__main__":

    print(tf.__version__)

    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()

    
    #plt.imshow(x_train[0], cmap = plt.cm.binary)
    #plt.show()

    # Normalizing the data
    x_train = tf.keras.utils.normalize(x_train, axis = 1)
    x_test = tf.keras.utils.normalize(x_test, axis = 1)

    #plt.imshow(x_train[0], cmap = plt.cm.binary)
    #plt.show()

    # Directed graph?
    model = tf.keras.models.Sequential()

    # Adding the layers
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
    # model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))

    # Output layer
    model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    model.fit(x_train, y_train, epochs = 3)

    val_loss, val_acc = model.evaluate(x_test, y_test)
    print(val_loss)
    print(val_acc)

    model.save('num_reader.model')

    predictions = model.predict(x_test)

    print(np.argmax(predictions[0]))

    plt.imshow(x_test[0], cmap = plt.cm.binary)
    plt.show()