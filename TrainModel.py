'''
Train the model for sleep classification
by Benjamin Thürer
'''
import tensorflow as tf
import numpy as np

n = 13  # 13 eeg channels
t = 256 * 30  # time points for each channels epoch
epoch_length = n * t

'''1d downsample to 50 nodes? Or downsample before'''
class CreateModel():
    '''Create model for deep learning'''
    def __init__(self, epoch_length):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Reshape((t, n), input_shape=(epoch_length,)))

        self.model.add(tf.keras.layers.Conv1D(8, 380, activation='relu', input_shape=(t, n), data_format='channels_last'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

        self.model.add(tf.keras.layers.Conv1D(8, 80, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=3))

        self.model.add(tf.keras.layers.Conv1D(12, 20, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(12, 5, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(16, 3, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Conv1D(16, 2, activation='relu'))
        self.model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

        self.model.add(tf.keras.layers.Flatten())

        self.model.add(tf.keras.layers.Dense(100, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())

        self.model.add(tf.keras.layers.Dense(50, activation='relu'))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.4))

        self.model.add(tf.keras.layers.Dense(6, activation='softmax'))

        optimizer = tf.keras.optimizers.Adam(lr=0.0004, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def getModel(self):
        return self.model

if __name__ == "__main__":
    n = 13  # 13 eeg channels
    t = 256 * 30  # time points for each channels epoch
    epoch_length = n * t
    model = CreateModel(epoch_length)
    model = model.getModel()
    test = np.random.randn(t*n,1)
    model.build(test)
    model.summary()


'''
class CreateModel():
    Create model for deep learning
    def __init__(self, epoch_length):
        self.model = tf.keras.Sequential()

        self.model.add(tf.keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.5))

        self.model.add(tf.keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.1))

        self.model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
        self.model.add(tf.keras.layers.BatchNormalization())
        self.model.add(tf.keras.layers.Dropout(0.2))

        self.model.add(tf.keras.layers.Dense(6, activation=tf.nn.softmax))

        optimizer = tf.keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def getModel(self):
        return self.model
'''