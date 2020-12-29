import numpy as np
import os
from scipy import linalg

file_dir = "/home/augus/ad/gae_pytorch/cora_output/"  # file directory
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

# there should be 6 with no anomaly files.
# normal feature, normal structure
# pca reduced scat feature, pca reduced scat structure
# fms reduced scat feature, fms reduced scat structure

# we now compare normal
print("compare normal")
temp_anomaly_list = []
key_1 = "normal"
key_2 = "FeatureDecoder"
key_3 = "na"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)

for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features =  temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file- temp_base)
    print(key_1, key_2, selected_anomaly_file, "encoder distance", norm_dis)


temp_anomaly_list = []
key_1 = "normal"
key_2 = "StructureDecoder"
key_3 = "na"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)

for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features =  temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file- temp_base)
    print(key_1, key_2, selected_anomaly_file, "encoder distance", norm_dis)
print()
print("compare reduced scat")
# we now compare scat
temp_anomaly_list = []
key_1 = "scat"
key_2 = "FeatureDecoder"
key_3 = "PCA"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file and key_3 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file and key_3 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)
for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features =  temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file- temp_base)
    print(key_1, key_2, key_3, selected_anomaly_file, "reduced scat distance", norm_dis)


temp_anomaly_list = []
key_1 = "scat"
key_2 = "FeatureDecoder"
key_3 = "FMS"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file and key_3 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file and key_3 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)
for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features =  temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file- temp_base)
    print(key_1, key_2, key_3, selected_anomaly_file, "reduced scat distance", norm_dis)


temp_anomaly_list = []
key_1 = "scat"
key_2 = "StructureDecoder"
key_3 = "PCA"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file and key_3 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file and key_3 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)
for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features = temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file - temp_base)
    print(key_1, key_2, key_3, selected_anomaly_file, "reduced scat distance", norm_dis)


temp_anomaly_list = []
key_1 = "scat"
key_2 = "StructureDecoder"
key_3 = "FMS"
for single_base_file in base_files_list:
    if key_1 in single_base_file and key_2 in single_base_file and key_3 in single_base_file:
        temp_base = np.load(single_base_file, allow_pickle=True)
for single_anomaly_file in anomaly_files_list:
    if key_1 in single_anomaly_file and key_2 in single_anomaly_file and key_3 in single_anomaly_file:
        temp_anomaly_list.append(single_anomaly_file)
for selected_anomaly_file in temp_anomaly_list:
    temp_single_anomaly_file = np.load(selected_anomaly_file, allow_pickle=True)
    num_nodes, num_features = temp_single_anomaly_file.shape
    for node in range(num_nodes):
        norm_dis = linalg.norm(temp_single_anomaly_file - temp_base)
    print(key_1, key_2, key_3, selected_anomaly_file, "reduced scat distance", norm_dis)