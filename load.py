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

# used log paser from here: https://lasio.readthedocs.io/en/latest
# read first las file
las = lasio.read("data/las/00a60e5cc262_TGS.las")
# las = lasio.read("data/las/0052442d0162_TGS.las")

# check existing curves and data shapes
print(las.curves)
print(las.data.shape)
a = las.curves

for curve in las.curves:
    print(f"{curve.mnemonic}: \ndata shape: {curve.data.shape} \n{str(curve.data)}\n ")

# store las as pandas dataframe for quick analysis
# 'DEPT' was converted to df.index automatically, add 'DEPT' column back later
df = las.df()
df["DEPT"] = df.index
df.describe()
plot_logs(df)

#%% convert las.curves info to df for better reading


def convert_curves2df(las):
    l = []
    for i in las.curves:
        l.append([i.mnemonic, i.descr])
    return pd.DataFrame(l)


#%% read all las file to df and store to hdf file format

las_data = pd.HDFStore("data/las_data.h5")
las_curves = pd.HDFStore("data/las_curves.h5")


iterator = count()
time_start = time.time()
for f in glob.glob("data/las/*.las"):
    f_name = f.split("/")[-1].split("\\")[-1][:-4]
    print(f"Loading {next(iterator)}th las file: \t\t{f_name}.las")
    las = lasio.read(f)
    df = las.df()

    if "DEPT" not in df.columns:
        df["DEPT"] = df.index

    las_data[f_name] = df
    las_curves[f_name] = convert_curves2df(las)

    # if next(iterator) > 2:
    #     break


las_data.close()
las_curves.close()


print(f"\nSuccessfully loaded total {next(iterator)} las files!")
print(f"Total run time: {time.time()-time_start: .2f} seconds")
