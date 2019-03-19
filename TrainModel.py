'''
Train the model for sleep classification
by Benjamin Thürer
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

''' matrix is too big for pre-allocation, therefore use separate load function to save memory'''
def load_data(hypnos_learn, i, D, files_learn, min_count):
    if hypnos_learn[i] not in D:
        D[hypnos_learn[i]] =1
        return np.loadtxt(f"{data_path}{files_learn[i]}"), i, hypnos_learn[i]
    elif hypnos_learn[i] in D and D[hypnos_learn[i]] <= min_count:
        D[hypnos_learn[i]] += 1
        return np.loadtxt(f"{data_path}{files_learn[i]}"), i, hypnos_learn[i]
    else:
        load_data(hypnos_learn, i+1, D, files_learn, min_count)

'''Create model for deep learning'''
def create_model(epoch_length):
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, input_shape=(epoch_length,), activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# load data for learning
data_path = "/home/benjamin/Benjamin/EEGdata_for_learning/"
files = os.listdir(data_path)
files.sort()  # might crash because too big
hypnos_learn = np.loadtxt(f"{data_path}{files[0]}")
files_learn = files[1:]

# load data for testing
data_path_test = "/home/benjamin/Benjamin/EEGdata_for_testing/"
files = os.listdir(data_path_test)
files.sort()  # might crash because too big
hypnos_test = np.loadtxt(f"{data_path_test}{files[0]}")
files_test = files[1:]

# get lowest amount across all sleep stages
_, counts = np.unique(hypnos_learn, return_counts=True)
min_count = min(counts)

# preallocate memory
n = 13  # 13 eeg channels
s_rate = 256
epoch_length = s_rate*30*n

# create model like this?
model = create_model(epoch_length)


''' next steps: use tf.dataset to load big amounts of data in batch'''
D = {}
i = 0
batch_size = 30
hypno_batch = np.zeros((batch_size))
data_learn = np.zeros((batch_size, epoch_length))
while i < files_learn.__len__():
    # load as many datasets as batch size
    n = 0
    while n < batch_size:
        data_learn[n][:], i, hypno_batch[n] = load_data(hypnos_learn, i, D, files_learn, min_count)
        i += 1
        n += 1

    # normalize data!!!
    # randomize data!!
    model.fit(data_learn, hypno_learn, epochs=5)
'''

import tensorflow as tf
from .pair_generator import PairGenerator
from .model import Inputs


class Dataset(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    label = 'same_person'

    def __init__(self, generator=PairGenerator()):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, pair_gen: PairGenerator):
        batch_size = 10
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(pair_gen.get_next_pair,
                                                 output_types={PairGenerator.person1: tf.string,
                                                               PairGenerator.person2: tf.string,
                                                               PairGenerator.label: tf.bool})
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        return Inputs(element[self.img1_resized],
                      element[self.img2_resized],
                      element[PairGenerator.label])

    def _read_image_and_resize(self, pair_element):
        target_size = [128, 128]
        # read images from disk
        img1_file = tf.read_file(pair_element[PairGenerator.person1])
        img2_file = tf.read_file(pair_element[PairGenerator.person2])
        img1 = tf.image.decode_image(img1_file)
        img2 = tf.image.decode_image(img2_file)

        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])
        img2.set_shape([None, None, 3])

        # resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        img2_resized = tf.image.resize_images(img2, target_size)

        pair_element[self.img1_resized] = img1_resized
        pair_element[self.img2_resized] = img2_resized
        pair_element[self.label] = tf.cast(pair_element[PairGenerator.label], tf.float32)

        return pair_element

RUN:

from recognizer.pair_generator import PairGenerator
from recognizer.tf_dataset import Dataset
from recognizer.model import Model
import tensorflow as tf
import pylab as plt
import numpy as np


def main():
    generator = PairGenerator()
    # print 2 outputs from our generator just to see that it works:
    iter = generator.get_next_pair()
    for i in range(2):
        print(next(iter))
    ds = Dataset(generator)
    model_input = ds.next_element
    model = Model(model_input)

    # train for 100 steps
    with tf.Session() as sess:
        # sanity test: plot out the first resized images and their label:
        (img1, img2, label) = sess.run([model_input.img1, 
                                        model_input.img2, 
                                        model_input.label])

        # img1 and img2 and label are BATCHES of images and labels. plot out the first one
        plt.subplot(2, 1, 1)
        plt.imshow(img1[0].astype(np.uint8))
        plt.subplot(2, 1, 2)
        plt.imshow(img2[0].astype(np.uint8))
        plt.title(f'label {label[0]}')
        plt.show()

        # intialize the model
        sess.run(tf.global_variables_initializer())
        # run 100 optimization steps
        for step in range(100):
            (_, current_loss) = sess.run([model.opt_step, 
                                          model.loss])
            print(f"step {step} log loss {current_loss}")


if __name__ == '__main__':
    main()


test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)

'''

'''
predictions = model.predict(test_images)
predictions[0]
'''


