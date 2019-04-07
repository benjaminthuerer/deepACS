'''
convert EEG data to learning and training datasets, where each epoch is saved as a .gz file
by Benjamin Thürer
'''
import os
import pyedflib
import numpy as np
from runstats import Statistics
import pandas as pd
from scipy import signal

# set paths for data and get all files in path
data_path = "/home/benjamin/Benjamin/DAVOS_data/"
data_hypno = f"{data_path}davos_run1_sleep_score_hypn_export_datasetnum"
data_data = f"{data_path}DAVOS_"
files_folder = os.listdir(data_path)
files_folder.sort()

# folder path for saving (this must be changed for testing)
save_path = "/home/benjamin/Benjamin/EEGdata_for_learning/"  # change for testing!

# loop through each file in folder. If .txt: read as hypo and read EDF
final_hg = []
stats = Statistics()

for file in files_folder:
    try:
        file.index(".txt")
    except:
        continue

    sb_id = file[-8:-4]
    if sb_id == "1089":
        continue  # because subject 1089 for testing! change == to != if save for testing (change save_path!)

    # load hypnogram and read EDF for specific EEG file
    print(f"\n\n Load data of subject nr: {sb_id} \n\n")
    hypno = open(f"{data_hypno}{sb_id}.txt", "r")
    f = pyedflib.EdfReader(f"{data_data}{sb_id}_(1).edf")

    # get hypnogram for this file
    hg = []
    [hg.append(x) for x in hypno]

    # include only EEG channels (first 13 data channels) and compute epoch_length
    n = 13
    signal_labels = f.getSignalLabels()
    s_rate = 256
    epoch_length = s_rate * 30  # for 30s epoch

    # linked mastoid for re-referencing
    M1 = signal_labels.index('M1')
    M2 = signal_labels.index('M2')
    linked_M = lambda x: (x[M1] - x[M2]) / 2  # compute linked mastoid reference

    # preallocate memory and get start
    data = np.zeros((n * epoch_length))
    start_d = epoch_length * 120  # exclude the first hour due to artifacts

    # just for printing percentage of progress
    perc = np.linspace(10, 100, 10)
    thresh = (f.getNSamples()[0] - start_d) // 10
    perc2 = start_d + thresh
    p = 0
    print("start looping through data and saving. This may take a while...")

    # loop through 30sec epochs of the data and save each epoch
    while f.getNSamples()[0] >= start_d + epoch_length:
        if int(hg[start_d // epoch_length][0]) != 8:  # as long as epoch is not an artifact (8)
            for i in np.arange(n):
                # add 30s for each channel into 'data'
                data[0 + i * epoch_length: i * epoch_length + epoch_length] = f.readSignal(i, start_d, epoch_length)

            df = pd.DataFrame(data.reshape(data.__len__() // n, n))
            df = pd.DataFrame(signal.resample(df, 30 * 64))  # downsample to 64 hz
            M_ref = linked_M(df)
            df = df.sub(M_ref, axis='rows')
            df.__delitem__(M1)
            df.__delitem__(M2)
            r, c = df.shape
            dataN = np.array(df).reshape(r*c)
            [stats.push(v) for v in dataN]

            np.savetxt(f"{save_path}learn_{sb_id}_{start_d}.gz", dataN)
            final_hg.append(int(hg[start_d // epoch_length][0]))
        start_d += epoch_length

        if start_d >= perc2:
            print(f"{perc[p]} % done")
            perc2 += thresh
            p += 1

# save the overall hypno of all files
np.savetxt(f"{save_path}hypnos.gz", final_hg)

data_min_max = [[stats.minimum()], [stats.maximum()]]
np.savetxt(f"{save_path}min_max.gz", data_min_max)

data_mean_std = [[stats.mean()], [stats.stddev()]]
np.savetxt(f"{save_path}sum_N.gz", data_mean_std)