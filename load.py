# this file load the each las data to dataframe and save as h5 format file
# the resulting file is the .h5 file in /data/ folder.

import glob
import pickle
import random
import time
from itertools import count
from plot import plot_logs
import lasio
import numpy as np
import pandas as pd

from util import read_las, process_las

# used log paser from here: https://lasio.readthedocs.io/en/latest
# read first las file
las = lasio.read("data/las/00a60e5cc262_TGS.las")


# check existing curves and data shapes
print(las.curves)
print(las.well)
print(las.data.shape)
a = las.curves

# for curve in las.curves:
#     print(f"{curve.mnemonic}: \ndata shape: {curve.data.shape} \n{str(curve.data)}\n ")

# store las as pandas dataframe for quick analysis
# 'DEPT' was converted to df.index automatically
df = las.df()
df.describe()
plot_logs(df)

#%% convert las.curves info to df for better reading
#las = lasio.read("data/las/00a60e5cc262_TGS.las")
las = "data/las/0052442d0162_TGS.las"

las = read_las(las)
print(las.df())
print(las.df_curvedata())
print(las.df_welldata())
print(las.get_mnemonic_unit())
print('lat and lon:', las.get_lat_lon())
print('start-stop depth:', las.get_start_stop())

#%% read all las file to df and store to hdf file format

las_data = pd.HDFStore("data/las_data.h5")
las_curves = pd.HDFStore("data/las_curves.h5")
cords = []


time_start = time.time()
count_ = 0
for f in glob.glob("data/las/*.las"):
    f_name = f.split("/")[-1].split("\\")[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_}th las file: \t\t{f_name}.las")
    las = read_las(f)
    
    las_data[f_name] = las.df()
    las_curves[f_name] = las.df_curvedata()

    cords.append([f_name]+las.get_lat_lon()+las.get_start_stop())

    count_ +=1

las_data.close()
las_curves.close()

cords = pd.DataFrame(cords, columns=['Well', 'Lat', 'Lon', 'STRT', 'STOP'])
cords.to_csv('data/cords.csv')

print(f'cords: \n{cords.head()}')
print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")




        
# test
df1 = process_las().keep_valid_DTSM_only(df)
df1

#%% read all las files to df, keep valid DTSM only, and store to hdf file format

las_data_DTSM = pd.HDFStore("data/las_data_DTSM.h5")
list_no_DTSM = []

time_start = time.time()
count_ = 0
for f in glob.glob("data/las/*.las"):
    f_name = f.split("/")[-1].split("\\")[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_}th las file: \t\t{f_name}.las")

    las = read_las(f)    
    df = las.df()

    try:
        las_data_DTSM[f_name] = process_las().keep_valid_DTSM_only(df)
    except:
        list_no_DTSM.append(f_name)

    count_ +=1

las_data_DTSM.close()

list_no_DTSM = pd.DataFrame(list_no_DTSM, columns=['las without DTSM'])
list_no_DTSM.to_csv('data/list_no_DTSM.csv', index=True)

print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")


# convert h5 to pickle
las_data_DTSM = pd.HDFStore("data/las_data_DTSM.h5")
las_data_DTSM_ = dict()
for key in las_data_DTSM.keys():
    las_data_DTSM_[key] = las_data_DTSM[key]
with open('data/las_data_DTSM.pickle', 'wb') as f:
    pickle.dump(las_data_DTSM_, f)

las_data_DTSM.close()