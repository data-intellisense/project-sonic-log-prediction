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

from plot import plot_logs_columns
from util import alias_dict, read_las, process_las, get_mnemonic, get_alias


#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format

path = f"data/leaderboard_1"

las_raw = dict()
las_data = dict()
las_data_DTSM = dict()
las_lat_lon = dict()


time_start = time.time()
count_ = 0
for f in glob.glob(f"{path}/*.las"):
    f_name = re.split("[\\\/]", f)[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_:0>3}th las file: \t{f_name}.las")

    las = read_las(f)    
    df = las.df()
    
    # save all data
    las_data[f_name] = df.copy()

    # save lat-lon
    las_lat_lon[f_name] = las.get_lat_lon()

    count_ +=1

    plot_logs_columns(df, 
                      well_name=f_name, 
                      plot_save_file_name=f_name, 
                      plot_save_path=path, 
                      plot_save_format=['png', 'html'])

# write las_data
with open(f"{path}/las_data.pickle", 'wb') as f:
    pickle.dump(las_data, f)

# write las_lat_lon
with open(f"{path}/las_lat_lon.pickle", 'wb') as f:
    pickle.dump(las_lat_lon, f)

print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")
