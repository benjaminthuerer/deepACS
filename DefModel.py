'''
Create model for sleep classification
by Benjamin Thürer
'''
import tensorflow as tf
import numpy as np


class CreateModel:
    """Create 1d CNN for deep learning"""
    def model_cnn(self, n, t, lrate, activate):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Reshape((t, n), input_shape=(n*t,)))

        self.model.add(tf.keras.layers.Conv1D(8, 7, activation=activate, input_shape=(t, n), data_format='channels_last'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=4))

        self.model.add(tf.keras.layers.Conv1D(12, 5, activation=activate))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

        self.model.add(tf.keras.layers.Conv1D(16, 3, activation=activate))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

        # self.model.add(tf.keras.layers.Conv1D(12, 3, activation=activate))
        # self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        # self.model.add(tf.keras.layers.Conv1D(14, 1, activation=activate))
        # self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(100, activation=activate))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        # self.model.add(tf.keras.layers.Dense(50, activation=activate))
        # self.model.add(tf.keras.layers.BatchNormalization())
        # self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return self.model

    def model_dense(self, n_dim, activate, lrate):
        self.model = tf.keras.models.Sequential()

        self.model.add(tf.keras.layers.Dense(200, activation=activate, input_shape=(n_dim,)))
        self.model.add(tf.keras.layers.BatchNormalization())
        # self.model.add(tf.keras.layers.Dropout(0.3))

        self.model.add(tf.keras.layers.Dense(120, activation=activate))
        # self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dense(60, activation=activate))
        # self.model.add(tf.keras.layers.Dropout(0.3))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dense(60, activation=activate))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(lr=lrate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # optimizer = tf.keras.optimizers.SGD(lrate)
        self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

        return self.model


if __name__ == "__main__":
    n = 12  # 11 eeg channels
    t = 64 * 30  # time points for each channels epoch
    n_dim = t*n
    activate = 'relu'
    lrate = 0.001
    model = CreateModel()
    model = model.model_dense(n_dim, activate, lrate)
    # test = np.random.randn(t * n, 1)
    # model.build(test)
    model.summary()
