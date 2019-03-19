'''
Train the model for sleep classification
by Benjamin Thürer
'''

import tensorflow as tf
from tensorflow import keras

class CreateModel():
    '''Create model for deep learning'''
    def __init__(self, epoch_length):
        self.model = keras.Sequential()

        self.model.add(keras.layers.Dense(280, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(keras.layers.BatchNormalization())
        self.model.add(keras.layers.Dropout(0.2))

        self.model.add(keras.layers.Dense(6, activation=tf.nn.softmax))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def getModel(self):
        return self.model

if __name__ == "__main__":
    n = 13  # 13 eeg channels
    s_rate = 256
    epoch_length = s_rate * 30 * n  # arbitrary length
    model = CreateModel(epoch_length)
    model = model.getModel()
    print(model)