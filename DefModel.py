'''
Create model for sleep classification
by Benjamin Thürer
'''
import tensorflow as tf
import numpy as np


class CreateModel:
    """Create 1d CNN for deep learning"""
    def __init__(self, n, t):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Reshape((t, n), input_shape=(n*t,)))

        self.model.add(tf.keras.layers.Conv1D(8, 9, activation='relu', input_shape=(t, n), data_format='channels_last'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(8, 7, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(12, 5, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(12, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(14, 1, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(50, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def getModel(self):
        return self.model


if __name__ == "__main__":
    n = 11  # 11 eeg channels
    t = 64 * 30  # time points for each channels epoch
    model = CreateModel(n, t)
    model = model.getModel()
    test = np.random.randn(t * n, 1)
    model.build(test)
    model.summary()
