#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 28 21:48:03 2020

@author: suyuhao
"""


import argparse
import numpy as np
import os
from scipy import linalg
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='amazon_elec_photo', help='type of dataset output.')
args = parser.parse_args()


print("start analyzing!")
file_dir =  "/Users/suyuhao/documents/ad/12_27/{}_encoder/".format(args.data)  # file directory
print("file dir:", file_dir)
key = "NoAnomaly"
os.chdir(file_dir)
all_list = os.listdir(file_dir)
base_count = 0
base_files_list = []
total_npy_list = []
for single_file in all_list:
    # first obtain all npy files
    if single_file.endswith('.npy'):
        # find base files, no anomaly
        total_npy_list.append(single_file)
        if key in single_file:
            base_files_list.append(single_file)
anomaly_files_list = list(set(total_npy_list) - set(base_files_list))

key_1 = ["normal", "scat"]
key_2 = ["FeatureDecoder", "StructureDecoder", "twodecoders"]
key_3 = ["FeatureAnomaly", "StructureAnomaly", "BothAnomaly"]

logname = f'/Users/suyuhao/documents/ad/12_27/log_norm_{args.data}.csv'
if not os.path.exists(logname):
    with open(logname, 'w') as logfile:
        logwriter = csv.writer(logfile, delimiter=',')
        logwriter.writerow([
            'key1', 'key2', 'key3',
            'key4','norm'
        ])
for firstKey in key_1:
    if firstKey == "normal":
        for secondKey in key_2:
            for single_base_file in base_files_list:
                if firstKey in single_base_file and secondKey in single_base_file:
                    temp_base = np.load(single_base_file, allow_pickle=True)
                    print(single_base_file)
            for single_anomaly_file in anomaly_files_list:
                for thirdKey in key_3:
                    if firstKey in single_anomaly_file and secondKey in single_anomaly_file and thirdKey in single_anomaly_file:
                        temp_single_anomaly_file = np.load(single_anomaly_file, allow_pickle=True)
                        norm_dis = linalg.norm(temp_single_anomaly_file - temp_base)
                        print(firstKey, secondKey, thirdKey, "na", norm_dis)
                        print()
                        with open(logname, 'a') as logfile:
                            logwriter = csv.writer(logfile, delimiter=',')
                            logwriter.writerow(
                                [firstKey, secondKey, thirdKey, 'na', norm_dis])


