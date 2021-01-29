#%% this file load the each las data to dataframe and save them
# the resulting files are in /data/ folder.

import glob
import pickle
import random
import time
from itertools import count
import lasio
import numpy as np
import pandas as pd
import re

#from plot import plot_logs_columns
from util2 import read_las, process_las, get_mnemonic, get_alias

#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format

las_raw = dict()
las_data = dict()
las_data_DTSM = dict()
las_lat_lon = dict()

list_no_DTSM = []
curve_info = []

time_start = time.time()
count_ = 0
for f in glob.glob("data/las/*.las"):
    f_name = re.split("[\\\/]", f)[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_:0>3}th las file: \t{f_name}.las")

    curve_info.append([f"{count_:0>3}", f_name])

    # save raw las into a dict
    las_raw[f_name] = open(f).read()

    las = read_las(f)    
    df = las.df()
    
    # save all data
    las_data[f_name] = df.copy()

    # save only DTSM valid data
    if 'DTSM' in df.columns.map(alias_dict):
        # save only DTSM valid data    
        las_data_DTSM[f_name] = process_las().keep_valid_DTSM_only(df).copy()      
    else:
        # save a list with las names without DTSM and raise warning
        print(f'{f_name}.las has no DTSM!')
        list_no_DTSM.append(f_name)

    # save lat-lon
    las_lat_lon[f_name] = las.get_lat_lon()

    count_ +=1

list_no_DTSM = pd.DataFrame(list_no_DTSM, columns=['las without DTSM'])
# only save if it's not empty
if len(list_no_DTSM)>=1:
    list_no_DTSM.to_csv('data/list_no_DTSM.csv', index=True)

curve_info = pd.DataFrame(curve_info, columns=['WellNo', 'WellName'])
curve_info.to_csv('data/curve_info.csv', index=False)

# write las_raw
with open('data/las_raw.pickle', 'wb') as f:
    pickle.dump(las_raw, f)

# write las_data
with open('data/las_data.pickle', 'wb') as f:
    pickle.dump(las_data, f)

# write las_data_DTSM
with open('data/las_data_DTSM.pickle', 'wb') as f:
    pickle.dump(las_data_DTSM, f)

# write las_data_DTSM
with open('data/las_lat_lon.pickle', 'wb') as f:
    pickle.dump(las_lat_lon, f)

print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")

