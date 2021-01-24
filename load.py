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
class read_las:
    def __init__(self, las):
        self.las=lasio.read(las)
        return None
    def df(self):        
        # if "DEPT" not in df_.columns:
        #     df_["DEPT"] = df_.index
        return self.las.df()
        
    def df_curvedata(self):
        l = []
        for i in self.las.curves:
            l.append([i.mnemonic, i.unit, i.descr])
        return pd.DataFrame(l, columns=['mnemonic', 'unit', 'description'])

    def get_mnemonic_unit(self):
        return self.df_curvedata()[['mnemonic', 'unit']]
    
    def get_curvedata(self, data_names=[]):
        df = self.df_curves()

    def df_welldata(self, valid_value_only=True):
        l = []
        for i in self.las.well:
            if valid_value_only and (i.value is not ''):
                l.append([i.mnemonic, i.unit, i.value, i.descr])
        return pd.DataFrame(l, columns=['mnemonic', 'unit', 'value', 'description'])

    def get_welldata(self, data_names=[]):
        df = self.df_welldata()
        return [df[df['mnemonic']==i]['value'].values[0] for i in data_names]

    def get_lat_lon(self,data_names=['SLAT','SLON']):        
        return self.get_welldata(data_names=data_names)

    def get_start_stop(self, data_names=['STRT', 'STOP']):
        return self.get_welldata(data_names=data_names)
    
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



#%% process las data, keeping only DTSM data

class process_las:
    def __init__(self):
        return None

    def renumber_columns(self, c):
        c = list(c.values)
        assert isinstance(c, list)
        c = c.copy()
        for i in c:
            if c.count(i)>1:
                ix = [j for j, val in enumerate(c) if val==i]
                for m,n in enumerate(ix):            
                    c[n] = c[n]+'_'+str(m)
        return c

    def keep_valid_DTSM_only(self, df=None, alias_dict=alias_dict):

        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        # convert all alias to desired mnemonics
        df.columns = df.columns.map(alias_dict)                

        # drop all columns whose mnemonics are not recognized, i.e. NaN
        df = df.loc[:,df.columns.notnull()]
        #df.columns = df.columns.fillna('to_drop')

        # drop columns with all na
        df = df.dropna(axis=1, how='all')

        # attach suffix number to duplicate column names
        df.columns = self.renumber_columns(df.columns)

        # make sure only one 'DTSM' exist
        try:
            list(df.columns.values).count('DTSM') <= 1
        except:
            print("More than one 'DTSM' curve exist!")        

        # drop all empty rows in 'DTSM' column if there is a 'DTSM' column
        try:
            # if 'DTSM' in df.columns:
            df = df.dropna(subset=['DTSM'])
            return df  
        except Exception as err:
            print("No 'DTSM' curve in this dataset, only performed converting alias to mnemonics and dropping un-recognizable curves!")
            
        
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