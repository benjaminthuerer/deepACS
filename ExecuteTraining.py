import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(patience=10)


def execute_train(n_dim, batch_size, data_path, files_learn, hypnos_learn, batch_size_test, data_path_test,
                  hypnos_test, files_test, epochs, model, data_mean, data_std, data_min, data_max):
    # predefine steps for training
    total_steps = hypnos_learn.__len__() // batch_size
    total_steps_test = hypnos_test.__len__() // batch_size_test

    def normalize_d(data_n, h):
        """min-max normalization: 0-1 range"""
        data_min = []
        [data_min.append(data_n[v, :, :].min()) for v in range(data_n.shape[0])]
        data_n = data_n.transpose() - np.array(data_min)
        data_max = []
        [data_max.append(data_n[v, :, :].max()) for v in range(data_n.shape[0])]
        data_n = data_n.transpose() / np.array(data_max)
        return [data_n, h]

        # """z-transform data"""
        # data_n = (data_n - data_mean) / data_std
        # return [data_n, h]

        # normalized PSD
        # return [data_n[:, :n_dim], h]

    def generator():
        """load learning data in chuncks to spare memory"""
        start = 0
        stop = batch_size
        while True:
            # files and hypnos come as nested list, therefor feed in loops
            yield np.array([np.loadtxt(f"{data_path}{ch[0]}") for ch in files_learn[start:stop]]), [s for s in
                                                                                          hypnos_learn[start:stop]]
            start, stop = start + batch_size, stop + batch_size

    def generator_test():
        """load testing / validation data in chuncks"""
        start = 0
        stop = batch_size_test
        while True:
            # files and hypnos come as nested list, therefor feed in loops
            yield [np.loadtxt(f"{data_path_test}{ch[0]}") for ch in files_test[start:stop]], [s for s in
                                                                                              hypnos_test[start:stop]]
            start, stop = start + batch_size_test, stop + batch_size_test

    # create Dataset with from_generator procedure and apply normalization
    dataset_learn = tf.data.Dataset.from_generator(generator, (tf.float32, tf.float32))
    dataset_learn = dataset_learn.map(lambda lx, lz: tf.py_func(normalize_d, [lx, lz], [tf.float32, tf.float32]))
    iter_learn = dataset_learn.make_one_shot_iterator()
    data, hypno = iter_learn.get_next()

    dataset_test = tf.data.Dataset.from_generator(generator_test, (tf.float32, tf.float32))
    dataset_test = dataset_test.map(lambda lx, lz: tf.py_func(normalize_d, [lx, lz], [tf.float32, tf.float32]))
    iter_test = dataset_test.make_one_shot_iterator()
    data_test, hypno_test = iter_test.get_next()

    # for saving:
    # saver = tf.train.Saver()

    # parallelise options CPU
    config = tf.ConfigProto(intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1,
                            allow_soft_placement=False)

    with tf.Session(config=config) as sess:
        # sess = tf.Session(config=config)
        d, v = sess.run([data, hypno])
        dt, vt = sess.run([data_test, hypno_test])
        # model.fit(d, v, batch_size=1, epochs=1, shuffle=False)
        # scores = model.evaluate(dt, vt, batch_size=1)
        # print(f"Model Accuracy: {scores[1]*100}")

        model.fit(d, v, batch_size=None, steps_per_epoch=total_steps, epochs=epochs, validation_data=(dt, vt),
                  validation_steps=total_steps_test, callbacks=[early_stop], shuffle=False)
        scores = model.evaluate(d, v, batch_size=batch_size)
        print(f"trained accuracy: {scores[1]*100}")
        scores = model.evaluate(dt, vt, batch_size=batch_size)
        print(f"test accuracy: {scores[1] * 100}")

        # save model:
        # model_json = model.to_json()
        # with open("model.json", "w") as json_file:
        #     json_file.write(model_json)
        # # serialize weights to HDF5
        # model.save_weights("model.h5")
        # print("Saved model to disk")

        # predict model:
        x = np.zeros(6)
        y = np.arange(0, 7)
        dp_corr = dict(zip(y, x))
        dp_false = dict(zip(y, x))

        # just for printing percentage of progress
        perc = np.linspace(10, 100, 10)
        thresh = files_test.__len__() // 10
        perc2 = thresh
        p = 0
        print("start looping through data for prediction. This may take a while...")

        for i, name in enumerate(files_test):
            data = np.loadtxt(f"{data_path_test}{name[0]}")
            data = data.reshape(1, data.__len__())
            hypno = hypnos_test[i]
            data, hypno = normalize_d(data, hypno)
            prediction = model.predict_classes(data, batch_size=1)
            if prediction == np.argmax(hypno):
                dp_corr[float(np.argmax(hypno))] += 1
            else:
                dp_false[float(np.argmax(hypno))] += 1

            if i >= perc2:
                print(f"{perc[p]} % done")
                perc2 += thresh
                p += 1

        for i in range(6):
            print(f"{i} corr: {round(dp_corr[i] / (dp_false[i] + dp_corr[i]) * 100)}% of {dp_false[i] + dp_corr[i]}")
