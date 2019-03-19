import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

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

# update files_learn to equal amounts of sleep stages (prevent overfitting to a specific stage)
D = {}
i = 0
new_hypno = []
new_files = []
while i < hypnos_learn.__len__():
    if hypnos_learn[i] not in D:
        D[hypnos_learn[i]] = 1
        new_hypno.append(hypnos_learn[i])
        new_files.append(files_learn[i])
    elif hypnos_learn[i] in D and D[hypnos_learn[i]] < min_count:
        D[hypnos_learn[i]] += 1
        new_hypno.append(hypnos_learn[i])
        new_files.append(files_learn[i])
    else:
        pass
    i += 1

files_learn = new_files
hypnos_learn = new_hypno

# define epoch_length
n = 13  # 13 eeg channels
s_rate = 256
epoch_length = s_rate*30*n

# shuffle data
c = np.c_[files_learn, hypnos_learn]
np.random.shuffle(c)
files_learn = c[:, :1]
hypnos_learn = c[:, 1:]

files_learn = files_learn.tolist()
hypnos_learn = hypnos_learn.tolist()

'''
Changes to be made:
- create model and feed data directly to model:
https://towardsdatascience.com/how-to-use-dataset-in-tensorflow-c758ef9e4428
'''


epochs = 1
batch_size = 2
def generator():
    start = 0
    stop = batch_size
    while True:
        # files and hypnos come as nested list, therefor feed in loops
        yield [np.loadtxt(f"{data_path}{x[0]}") for x in files_learn[start:stop]], [v[0] for v in hypnos_learn[start:stop]]
        start, stop = start + batch_size, stop + batch_size

dataset = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
iter = dataset.make_one_shot_iterator()
data, hypno = iter.get_next()

with tf.Session() as sess:
    print(sess.run([data, hypno]))
    print(sess.run([data, hypno]))
    sess.close()