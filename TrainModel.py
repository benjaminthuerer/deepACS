"""
Problem: overfitting; Changes to be made:
- switch to LSTM (reshape: time, 1 look back, 5 channels)
- min-max for each epoch!
- more data
- test pipeline with MNIST?
- more layers
"""

import os
import numpy as np
import tensorflow as tf
from DefModel import CreateModel
from ExecuteTraining import execute_train


# set number of epochs and batch size
epochs = 2
batch_size = 60
batch_size_test = 60

# define number of EEG channels and s_rate
n = 5  # 12 eeg channels
s_rate = 32
t = s_rate * 30
n_dim = 513  # for PSD only

# load and sort data for learning
data_path = "/home/benjamin/Benjamin/EEGdata_for_learning/"
files = os.listdir(data_path)
files.sort()
hypnos_learn = np.loadtxt(f"{data_path}hypnos.gz")
min_max = np.loadtxt(f"{data_path}min_max.gz")
sum_N = np.loadtxt(f"{data_path}sum_N.gz")
del files[files.index("hypnos.gz")]
del files[files.index("min_max.gz")]
del files[files.index("sum_N.gz")]
files_learn = files[:]

# load and sort data for testing
data_path_test = "/home/benjamin/Benjamin/EEGdata_for_testing/"
files = os.listdir(data_path_test)
files.sort()
hypnos_test = np.loadtxt(f"{data_path_test}hypnos.gz")
del files[files.index("hypnos.gz")]
del files[files.index("min_max.gz")]
del files[files.index("sum_N.gz")]
files_test = files

# get lowest count across all sleep stages (ensure same amount of training data for each condition)
_, counts = np.unique(hypnos_learn, return_counts=True)
min_count = min(counts)  # min count 1368 (6*1368=8208 not much data!) || now 2165 (...=12990)

# update files_learn to equal amounts of sleep stages
D = {}
i = 0
new_hypno = []
new_files = []

"""this disturbs the order, right?!"""
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

"""not shuffle for LSTM??"""
# shuffle learning data
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

# mean, std, min, max:
data_mean = sum_N[0]
data_std = sum_N[1]
data_min = min_max[0]
data_max = min_max[1]

# convert to categoricals
_ = []
[_.append(int(float(v[0]))) for v in hypnos_learn]
hypnos_learn = _
# hypnos_learn = tf.keras.utils.to_categorical(_)

_ = []
[_.append(int(float(v[0]))) for v in hypnos_test]
hypnos_test = _
# hypnos_test = tf.keras.utils.to_categorical(_)

learning_rates = [0.001]
activate = 'relu'

for lrate in learning_rates:
    # create model from DefModel.py
    model = CreateModel()

    model = model.model_lstm(n, t, lrate, activate, batch_size)
    # model = model.model_cnn(n, t, lrate, activate)
    # model = model.model_dense(n*t, activate, lrate)

    # execute training
    execute_train(n_dim, batch_size, data_path, files_learn, hypnos_learn, batch_size_test, data_path_test,
                  hypnos_test, files_test, epochs, model, data_mean, data_std, data_min, data_max)
