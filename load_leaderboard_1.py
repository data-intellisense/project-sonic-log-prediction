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

# modify alias_dict
alias_dict_test = dict(alias_dict, **dict(SPHI_LS='NPHI'))
alias_dict_test['SPHI_LS']

#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format

path = f"data/leaderboard_1"

las_raw = dict()
las_data = dict()
las_data_TEST = dict()
las_lat_lon = dict()
test_file_list = []

time_start = time.time()
count_ = 1
for f in glob.glob(f"{path}/*.las"):
    f_name = re.split("[\\\/]", f)[-1][:-4]
    f_name = f"{count_:0>3}-{f_name}"
    print(f"Loading {count_:0>3}th las file: \t{f_name}.las")
    test_file_list.append(f_name)

    las = read_las(f)    
    df = las.df()
    
    # save all data
    las_data[f_name] = df.copy()

    # save lat-lon
    las_lat_lon[f_name] = las.get_lat_lon()

    count_ +=1

    plot_logs_columns(df, 
                      well_name=f'{f_name} raw data', 
                      plot_save_file_name=f_name, 
                      plot_save_path=path, 
                      plot_save_format=['png', 'html'],
                      alias_dict=alias_dict_test,
                      )

# write file names
test_file_list = pd.DataFrame(test_file_list, columns = ['WellName'])
test_file_list.to_csv(f"{path}/test_file_list.csv")

# write las_data
with open(f"{path}/las_data.pickle", 'wb') as f:
    pickle.dump(las_data, f)

# write las_lat_lon
with open(f"{path}/las_lat_lon.pickle", 'wb') as f:
    pickle.dump(las_lat_lon, f)

print(f"\nSuccessfully loaded total {count_+1} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")


#%% pick curves

leaderboard_1_inputs = pd.read_csv('data/leaderboard_1/leaderboard_1_inputs.csv')

# read las_data
with open(f"{path}/las_data.pickle", 'rb') as f:
    las_data = pickle.load(f)

# create a new dict
las_data_TEST = dict()

# remove the undesired curves
temp = leaderboard_1_inputs[['WellName','Curves to remove']]
for ix, WellName, curves_to_remove in temp.itertuples():

    curves_to_remove = [i.strip() for i in str(curves_to_remove).split(',')]
    print(WellName, 'removing', curves_to_remove[:2], '...')

    # check if all 'curves_to_remove' are in columns names, then drop curves
    if all([i in las_data[WellName].columns for i in curves_to_remove]):
        las_data_TEST[WellName] = las_data[WellName][las_data[WellName].columns.difference(curves_to_remove)]           
        remaining_mnemonics = [get_mnemonic(i) for i in las_data_TEST[WellName].columns]
        for i in curves_to_remove:
            if (get_mnemonic(i) not in remaining_mnemonics) and (i != 'AHFCO60'):
                print(f'\tRemoving {i} from data, while {remaining_mnemonics} does not have !')
    else:
        las_data_TEST[WellName] = las_data[WellName]
        if curves_to_remove != ['nan']:
            print(f"\tNot all {curves_to_remove} are in {WellName} columns. No curves are removed!")    


print('*'*90)

for key in las_data_TEST.keys():
    plot_logs_columns(las_data_TEST[key], 
        well_name=f'{key}-test-data', 
        plot_save_file_name=f'{key}-test-data', 
        plot_save_path=path, 
        plot_save_format=['png', 'html'],
        alias_dict=alias_dict_test,
        )

        