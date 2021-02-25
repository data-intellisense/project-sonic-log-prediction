#%% this file load the each las data to dataframe and save them
# the resulting files are in /data/ folder.

import glob
import pickle
import time
import numpy as np
import pandas as pd
import re

from plot import plot_logs_columns
from util import read_las, process_las, get_mnemonic, to_pkl, read_pkl
from load_pickle import alias_dict


#%% read all las files to df, keep all and valid DTSM only, and store to pickle file format

path = f"data/leaderboard_3"

las_data = dict()  # raw df in las files
las_lat_lon_TEST = dict()  # lat and lon of test wells
las_depth_TEST = dict()  # STRT and STOP depth of test wells
las_list_TEST = []  # name of the test files

time_start = time.time()
count_ = 11
for f in glob.glob(f"{path}/*.las"):

    # get the file name w/o extension
    WellName = re.split("[\\\/.]", f)[-2]
    WellName = f"{count_:0>3}-{WellName}"
    print(f"Loading {count_:0>3}th las file: \t{WellName}.las")
    count_ += 1

    # save the WellName for future use
    las_list_TEST.append(WellName)

    # get raw df and save to a dict
    las = read_las(f)
    df = las.df()
    las_data[WellName] = df.copy()

    # get lat-lon and save to a dict
    las_lat_lon_TEST[WellName] = las.get_lat_lon()

    # get depth and save to a dict
    las_depth_TEST[WellName] = [df.index.min(), df.index.max()]

    # plot the raw logs for visual inspection
    plot_logs_columns(
        df,
        well_name=f"{WellName}-raw-data",
        plot_show=False,
        plot_save_file_name=WellName,
        plot_save_path=path,
        plot_save_format=["png"],
        alias_dict=alias_dict,
    )

# write file names
las_list_TEST = pd.DataFrame(las_list_TEST, columns=["WellName"])
las_list_TEST.to_csv(f"{path}/las_list_TEST.csv")

# write las_data
to_pkl(las_data, f"{path}/las_data.pickle")

# write las_lat_lon_TEST
to_pkl(las_lat_lon_TEST, f"{path}/las_lat_lon_TEST.pickle")

# write las_depth_TEST
to_pkl(las_depth_TEST, f"{path}/las_depth_TEST.pickle")

print(f"\nSuccessfully loaded total {len(las_data)} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")

#%% pick the curves from test data, requires log_QC_input.csv

log_QC_input = pd.read_csv(f"{path}/log_QC_input_3.csv")

# read las_data
las_data = read_pkl(f"{path}/las_data.pickle")
las_data_TEST = dict()  # cleaned df from las_data

# remove the undesired curves
temp = log_QC_input[["WellName", "Curves to remove"]]
for ix, WellName, curves_to_remove in temp.itertuples():

    curves_to_remove = [i.strip() for i in str(curves_to_remove).split(",")]
    print(WellName, "removing", curves_to_remove[:2], "...")

    # check if all 'curves_to_remove' are in columns names, if YES then drop these curves
    if all([i in las_data[WellName].columns for i in curves_to_remove]):
        las_data_TEST[WellName] = las_data[WellName][
            las_data[WellName].columns.difference(curves_to_remove)
        ]
        remaining_mnemonics = [
            get_mnemonic(i, alias_dict=alias_dict)
            for i in las_data_TEST[WellName].columns
        ]

        # make sure not to remove curves by accident
        for i in curves_to_remove:
            if get_mnemonic(i, alias_dict=alias_dict) not in remaining_mnemonics:
                print(
                    f"\tRemoving {i} from data, while {remaining_mnemonics} does not have !"
                )
    else:
        las_data_TEST[WellName] = las_data[WellName]
        if curves_to_remove != ["nan"]:
            print(
                f"\tNot all {curves_to_remove} are in {WellName} columns. No curves are removed!"
            )

    # plot the QC'd logs
    for key in las_data_TEST.keys():
        plot_logs_columns(
            df=las_data_TEST[key],
            well_name=f"{key}-test-data",
            plot_show=False,
            plot_save_file_name=f"{key}-test-data",
            plot_save_path=path,
            plot_save_format=["png", "html"],
            alias_dict=alias_dict,
        )

# write las_data
to_pkl(las_data_TEST, f"{path}/las_data_TEST.pickle")

print("*" * 90)
print("Congratulations! Loaded data successfully!")
