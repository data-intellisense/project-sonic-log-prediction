#%% this file load the each las data to dataframe and save as h5 format file
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

from plot import plot_logs_columns
from util import alias_dict, read_las, process_las, get_mnemonic, get_alias

#%% used log paser from here: https://lasio.readthedocs.io/en/latest
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

#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format

las_data = dict()
las_data_DTSM = dict()
list_no_DTSM = []
curve_info = []

time_start = time.time()
count_ = 0
for f in glob.glob("data/las/*.las"):
    f_name = f.split("/")[-1].split("\\")[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_:0>3}th las file: \t{f_name}.las")

    curve_info.append([f"{count_:0>3}", f_name])

    las = read_las(f)    
    df = las.df()
    
    # save all data
    las_data[f_name] = df.copy()

    if 'DTSM' in df.columns.map(alias_dict):
        # save only DTSM valid data    
        las_data_DTSM[f_name] = process_las().keep_valid_DTSM_only(df).copy()      
    else:
        # save a list with las names without DTSM and raise warning
        print(f'{f_name}.las has no DTSM!')
        list_no_DTSM.append(f_name)
        
    count_ +=1


list_no_DTSM = pd.DataFrame(list_no_DTSM, columns=['las without DTSM'])
list_no_DTSM.to_csv('data/list_no_DTSM.csv', index=True)

curve_info = pd.DataFrame(curve_info, columns=['WellNo', 'WellName'])
curve_info.to_csv('data/curve_info.csv', index=False)

# save las_data
with open('data/las_data.pickle', 'wb') as f:
    pickle.dump(las_data, f)

# save las_data_DTSM
with open('data/las_data_DTSM.pickle', 'wb') as f:
    pickle.dump(las_data_DTSM, f)

print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")

# check if nan columns dropped
key = '001-00a60e5cc262_TGS'
a = las_data_DTSM[key]
a[(a.index>5500) & (a.index<8000)]

plot_logs_columns(a, plot_show=True, well_name=key)


