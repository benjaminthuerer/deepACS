'''
Problem: overfitting; Changes to be made:
- remove mean of whole training set!
- validation in training
- lower learning rate / regularization
- change layers (CNN for 1d data? or make 2d for channels?)
- RNN more sense?
'''

import os
import numpy as np
import tensorflow as tf
from TrainModel import CreateModel

epochs = 100
batch_size = 32
batch_size_test = 32

# define epoch_length
n = 13  # 13 eeg channels
s_rate = 256
epoch_length = s_rate * 30 * n

# load data for learning
data_path = "/home/benjamin/Benjamin/EEGdata_for_learning/"
files = os.listdir(data_path)
files.sort()  # might crash because too big
hypnos_learn = np.loadtxt(f"{data_path}hypnos.gz")
min_max = np.loadtxt(f"{data_path}min_max.gz")
sum_N = np.loadtxt(f"{data_path}sum_N.gz")
del files[files.index("hypnos.gz")]
del files[files.index("min_max.gz")]
del files[files.index("sum_N.gz")]
files_learn = files

# load data for testing
data_path_test = "/home/benjamin/Benjamin/EEGdata_for_testing/"
files = os.listdir(data_path_test)
files.sort()  # might crash because too big
hypnos_test = np.loadtxt(f"{data_path_test}hypnos.gz")
del files[files.index("hypnos.gz")]
files_test = files

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

# shuffle data
c = np.c_[files_learn, hypnos_learn]
np.random.shuffle(c)
files_learn = c[:, :1]
hypnos_learn = c[:, 1:]

files_learn = files_learn.tolist()
hypnos_learn = hypnos_learn.tolist()

# shuffle test data
c = np.c_[files_test, hypnos_test]
np.random.shuffle(c)
files_test = c[:, :1]
hypnos_test = c[:, 1:]

files_test = files_test.tolist()
hypnos_test = hypnos_test.tolist()

model = CreateModel(epoch_length)
model = model.getModel()

total_steps = hypnos_learn.__len__() // batch_size
total_steps_test = hypnos_test.__len__() // batch_size_test

data_mean = sum_N[0]
data_std = sum_N[1]


def normalize_d(data_n, h):
    data_n = (data_n - data_mean) / data_std
    return [data_n, h]


def generator():
    start = 0
    stop = batch_size
    while True:
        # files and hypnos come as nested list, therefor feed in loops
        yield [np.loadtxt(f"{data_path}{x[0]}") for x in files_learn[start:stop]], [s[0] for s in
                                                                                    hypnos_learn[start:stop]]
        start, stop = start + batch_size, stop + batch_size


def generator_test():
    start = 0
    stop = batch_size_test
    while True:
        # files and hypnos come as nested list, therefor feed in loops
        yield [np.loadtxt(f"{data_path_test}{x[0]}") for x in files_test[start:stop]], [s[0] for s in
                    hypnos_test[start:stop]]
        start, stop = start + batch_size_test, stop + batch_size_test


dataset_learn = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
dataset_learn = dataset_learn.map(lambda lx, lz: tf.py_func(normalize_d, [lx, lz], [tf.float32, tf.float32]))
iter = dataset_learn.make_one_shot_iterator()
data, hypno = iter.get_next()

'''check for accuracy with evaluate'''
dataset_test = tf.data.Dataset.from_generator(generator_test, (tf.float32, tf.float32))
dataset_test = dataset_test.map(lambda lx, lz: tf.py_func(normalize_d, [lx, lz], [tf.float32, tf.float32]))
iter = dataset_test.make_one_shot_iterator()
data_test, hypno_test = iter.get_next()

saver = tf.train.Saver()
config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=False)

sess = tf.Session(config=config)
d, v = sess.run([data, hypno])
dt, vt = sess.run([data_test, hypno_test])
model.fit(d, v, batch_size=None, steps_per_epoch=total_steps, epochs=epochs, validation_data=(dt, vt), validation_steps=total_steps_test)
sess.close()

#  iter = dataset_test.make_one_shot_iterator()
#  data_test, hypno_test = iter.get_next()
#  sess = tf.Session(config=config)
#  dt, vt = sess.run([data_test, hypno_test])
#  loss, acc = model.evaluate(dt, vt, batch_size=None, steps=total_steps_test)
#  print(f"\n\n############### \nloss: {loss}, accuracy: {acc} \n###############")

#  saver.save(sess, '/home/benjamin/PycharmProjects/DeepLearningNew/deepASC_model')
#  sess.close()
'''
set up EC2 cloud with jupyter:
https://hackernoon.com/keras-with-gpu-on-amazon-ec2-a-step-by-step-instruction-4f90364e49ac


with tf.Session() as sess:
    latest = tf.train.im
    new_saver = tf.train.import_meta_graph(sess, "")
    dt, vt = sess.run([data_test, hypno_test])
    loss, acc = model.evaluate(dt, vt, batch_size=None, steps=total_steps_test)
    print(f"loss: {loss}, accuracy: {acc}")
    sess.close()
'''
'''
with tf.Session() as sess:
    d, v = sess.run([data, hypno])
    model.fit(d, v, batch_size=None, steps_per_epoch=total_steps, epochs=epochs, callbacks=[cp_callback])
    sess.close()

latest_model = tf.train.latest_checkpoint(checkpoint_dir)
with tf.Session() as sess:
    dt, vt = sess.run([data_test, hypno_test])
    model = CreateModel(epoch_length)
    model.load_weights(latest_model)
    loss, acc = model.evaluate(dt, vt, batch_size=None, steps=total_steps_test)
    print(f"loss: {loss}, accuracy: {acc}")
    sess.close()


https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607
'''