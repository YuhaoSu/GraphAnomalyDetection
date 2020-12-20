import pandas as pd
import numpy as np
import os

file_dir = "/Users/suyuhao/Documents/AD/11_11_&_12_18_summarized_results/ms_academic_cs_output/"  # file directory

all_csv_list = os.listdir(file_dir)
count = 0
os.chdir(file_dir)
for single_csv in all_csv_list:
    count = count+1
    if count < 10:
        os.rename(single_csv, str(0)+str(count)+single_csv[0:-4]+".csv")
    else:
        os.rename(single_csv, str(count)+single_csv[0:-4]+".csv")

all_csv_list = os.listdir(file_dir)
all_csv_list.sort(key=lambda x: int(x[0:2]))

for single_csv in all_csv_list:
    print(single_csv)
    single_data_frame = pd.read_csv(os.path.join(file_dir, single_csv)).iloc[-1:]
    print(single_data_frame.info())
    if single_csv == all_csv_list[0]:
        all_data_frame = single_data_frame
    else:
        all_data_frame = pd.concat([all_data_frame, single_data_frame])
all_data_frame.csv_path = file_dir + "ms_academic_cs_output_all.csv"
all_data_frame.to_csv(all_data_frame.csv_path)
