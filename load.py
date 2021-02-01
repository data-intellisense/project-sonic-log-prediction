#%% this file load the each las data to dataframe and save them
# the resulting files are in /data/ folder.

import glob
import pickle
import random
import time
from itertools import count
from plot import plot_logs
import lasio
import numpy as np
import pandas as pd
import re

from plot import plot_logs_columns
from util import alias_dict, read_las, process_las, get_mnemonic, get_alias


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
# curve_info.to_csv('data/curve_info.csv', index=False)

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

if __name__ == '__main__':
    # check if nan columns dropped
    key = '001-00a60e5cc262_TGS'
    a = las_data_DTSM[key]
    print(a[(a.index>5500) & (a.index<8000)])

    # plot_logs_columns(a, plot_show=True, well_name=key)

#%% QC curves, remove unnecessary curves

curve_info_to_QC = pd.read_csv('data/curve_info_to_QC.csv')

# read las_data_DTSM
with open('data/las_data_DTSM.pickle', 'rb') as f:
    las_data_DTSM = pickle.load(f)

# create a new dict
las_data_DTSM_QC = dict()

# remove the undesired curves
temp = curve_info_to_QC[['WellName','Curves to remove', 'drop_las']]
for ix, WellName, curves_to_remove, drop_las in temp.itertuples():
    if pd.isnull(drop_las):
        curves_to_remove = [i.strip() for i in str(curves_to_remove).split(',')]
        print(WellName, 'removing', curves_to_remove[:2], '...')

        # check if all 'curves_to_remove' are in columns names, then drop curves
        if all([i in las_data_DTSM[WellName].columns for i in curves_to_remove]):
            las_data_DTSM_QC[WellName] = las_data_DTSM[WellName][las_data_DTSM[WellName].columns.difference(curves_to_remove)]           
            remaining_mnemonics = [get_mnemonic(i) for i in las_data_DTSM_QC[WellName].columns]
            for i in curves_to_remove:
                if (get_mnemonic(i) not in remaining_mnemonics) and (i != 'AHFCO60'):
                    print(f'\tRemoving {i} from data, while {remaining_mnemonics} does not have !')
        else:
            las_data_DTSM_QC[WellName] = las_data_DTSM[WellName]
            if curves_to_remove != ['nan']:
                print(f"\tNot all {curves_to_remove} are in {WellName} columns. No curves are removed!")
        


print('*'*90)

temp = curve_info_to_QC[['WellName','Top Depth', 'Btm Depth']]
for ix, WellName, top_depth, btm_depth in temp.itertuples():
    if not pd.isnull(top_depth):
        las_data_DTSM_QC[WellName] = las_data_DTSM_QC[WellName][las_data_DTSM_QC[WellName].index>=top_depth]
    if not pd.isnull(btm_depth):
        las_data_DTSM_QC[WellName] = las_data_DTSM_QC[WellName][las_data_DTSM_QC[WellName].index<=btm_depth]

print('The new number of las with DTSM is:', len(las_data_DTSM_QC))

for WellName in las_data_DTSM_QC.keys():
    las_data_DTSM_QC[WellName] = las_data_DTSM_QC[WellName].dropna(axis=1, how='all')
    # print(WellName,':dropping all na columns')            

# write las_data_DTSM
with open('data/las_data_DTSM_QC.pickle', 'wb') as f:
    pickle.dump(las_data_DTSM_QC, f)
