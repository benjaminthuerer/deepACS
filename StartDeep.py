'''
convert EEG data to learning dataset
by Benjamin Thürer
'''
from typing import List, Any
import os
import pyedflib
import numpy as np
import matplotlib.pyplot as plt
import time

# set paths for data and get all files in path
data_path = "/home/benjamin/Benjamin/DAVOS_data/"
data_hypno = f"{data_path}davos_run1_sleep_score_hypn_export_datasetnum"
data_data = f"{data_path}DAVOS_"
files_folder = os.listdir(data_path)
files_folder.sort()

# loop through each file in folder. If .txt read as hypo and read EDF
final_hg = []
for file in files_folder:
    try:
        file.index(".txt")
    except:
        continue

    sb_id = file[-8:-4]
    if sb_id == "1089":
        continue  # because subject 1089 for testing! change == to != if save for testing (change save folders!!!)

    print(f"\n\n Load data of subject nr: {sb_id} \n\n")
    hypno = open(f"{data_hypno}{sb_id}.txt", "r")

    # Read EDF of specific EEG
    f = pyedflib.EdfReader(f"{data_data}{sb_id}_(1).edf")


    # get hypnogram for this file
    hg = []
    [hg.append(x) for x in hypno]

    n = 13  # 13 because afterwards are not EEG channels anymore (what about EMG??)
    signal_labels = f.getSignalLabels()
    s_rate = 256
    epoch_length = s_rate*30  # for 30s epoch

    # preallocate memory and get start
    data = np.zeros((n*epoch_length))
    start_d = epoch_length*120  # exclude the first hour due to artifacts

    # just for printing percentage of progress
    perc = np.linspace(10, 100, 10)
    thresh = (f.getNSamples()[0] - start_d)//10
    perc2 = start_d + thresh
    p = 0
    print("start looping through data and saving. This may take a while...")

    saves = 0
    # loop through 30sec epochs of the data and save each epoch
    while f.getNSamples()[0] >= start_d + epoch_length:
        if int(hg[start_d//epoch_length][0]) != 8:  # as long as epoch is not an artifact (8)
            for i in np.arange(n):
                # add 30s for each channel into 'data'
                data[0+i*epoch_length:i*epoch_length+epoch_length] = f.readSignal(i, start_d, epoch_length)

            np.savetxt(f"/home/benjamin/Benjamin/EEGdata_for_learning/learn_{sb_id}_{start_d}.gz", data)
            saves += 1
            final_hg.append(int(hg[start_d//epoch_length][0]))
        start_d += epoch_length

        if start_d >= perc2:
            print(f"{perc[p]} % done")
            perc2 += thresh
            p += 1

# save the overall hypno of all files
np.savetxt("/home/benjamin/Benjamin/EEGdata_for_learning/hypnos.gz", final_hg)