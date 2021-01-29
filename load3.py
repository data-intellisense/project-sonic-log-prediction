#%% this module is to used test different models
import glob
import pickle
import random
import time
from itertools import count
import lasio
import numpy as np
import pandas as pd
import re
import pathlib 


path = pathlib.Path(__file__).parent

# get the alias_dict, required
with open(f'{path}/data/alias_dict.pickle', 'rb') as f:
    alias_dict = pickle.load(f)


# given a mnemonic, find all of its alias
def get_alias(mnemonic, alias_dict=alias_dict):
    alias_dict = alias_dict or {}
    return [k for k, v in alias_dict.items() if mnemonic in v]

# given a alias, find its corresponding one and only mnemonic
# return a mnemonis if found, or else ''
def get_mnemonic(alias = None, alias_dict=alias_dict):
    alias_dict = alias_dict or {}
    try:
        return [v for k, v in alias_dict.items() if alias == k][0]
    except:
        return ''


#%% read las, return curves and data etc.

class read_las:
    def __init__(self, las):
        self.las=lasio.read(las)
        return None
        
    def df(self):        
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
            if valid_value_only and (i.value != ''):
                l.append([i.mnemonic, i.unit, i.value, i.descr])
        return pd.DataFrame(l, columns=['mnemonic', 'unit', 'value', 'description'])

    def get_welldata(self, data_names=[]):
        df = self.df_welldata()
        return [df[df['mnemonic']==i]['value'].values[0] for i in data_names]

    def get_lat_lon(self,data_names=['SLAT','SLON']):        
        return self.get_welldata(data_names=data_names)

    def get_start_stop(self, data_names=['STRT', 'STOP']):
        return self.get_welldata(data_names=data_names) 


class process_las:
    def __init__(self):         
        return None

    def _renumber_columns(self, c):
        c = list(c.values)
        assert isinstance(c, list)
        c = c.copy()
        for i in c:
            if c.count(i)>1:
                ix = [j for j, val in enumerate(c) if val==i]
                for m,n in enumerate(ix):            
                    c[n] = c[n]+'_'+str(m)
        return c

    def despike(self, df=None, cols=None, window_size=3):
        '''
        df should be a dataframe, will despike all or selected columns
        '''
        assert isinstance(df, pd.DataFrame)
        assert window_size%2==1, "Median filter window size must be odd."

        if cols is not None:
            assert isinstance(cols, list), 'cols should be a list columns names of df'
            assert all([i in df.columns for i in cols]), 'cols should be a list columns names of df'            
        else:
            cols = df.columns
        
        for col in cols:
            df[col] = medfilt(df[col].values, kernel_size=window_size)

        return df

    def keep_valid_DTSM_only(self, df=None, alias_dict=alias_dict):
        '''
        return a df with rows of valid DTSM data
        '''    
        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

 
        # make sure only one 'DTSM' exist
        if list(df.columns.values).count('DTSM') > 1:
            print("More than one 'DTSM' curve exist!")

        if 'DTSM' not in df.columns:
            print('DTSM not present in this las file!')

        # drop all empty rows in 'DTSM' column if there is a 'DTSM' column
        # if 'DTSM' in df.columns, dropp all rows with nan in DTSM column
        df = df.dropna(subset=['DTSM'])

        # drop other columns with all na
        df = df.dropna(axis=1, how='all')

        return df  
       
        
    def get_df_by_mnemonics(self, df=None, target_mnemonics=None, strict_input_output=True):
        '''
        useage: get a cleaned dataframe by given mnemonics,
        target_mnemonics: a list of legal mnemonics,
        strict_input_output: if true, only output a df only if all target mnemonics are found in las, output None otherwise
            if false, will output a df with all possible mnemonis found in las
        '''

        df = df.copy()
        # check required parameters
        if target_mnemonics is None:
            print("Target mnemonics are required as 'get_df_by_mnemonics' function input!")
            return None
        else:
            assert isinstance(target_mnemonics, list)
            assert len(target_mnemonics)>=1
            assert all([i in alias_dict.values() for i in target_mnemonics]), \
                f'Mnemonics should be in the list of {np.unique(list(alias_dict.values()))}'

        # find alias in df for target mnemonics
        alias = dict()
        for m in target_mnemonics:
            alias[m] = []
            for col in df.columns:
                n = get_mnemonic(col, alias_dict=alias_dict)
                if (m==n):            
                    alias[m].append(col)

        df_ = None
        df_cols = []

        # key: target mnenomics; value: alias in df
        for key, value in alias.items():
            if len(value)==1:
                temp = df[value].values.reshape(-1,1)
                df_cols.append(key)
            elif len(value)>1:
                temp = df[value].mean(axis=1).values.reshape(-1,1)
                df_cols.append(key)
            elif len(value)==0: # return index if no such column
                # print(f'\tNo corresponding alias for {key}!')
                continue

            if df_ is None:
                df_ = temp
            else:
                df_ = np.c_[df_, temp]

        
        df_ = pd.DataFrame(df_, columns=df_cols)
        df_.index = df.index

        # dropped rows with na in DTSM column
        try:
            df_ = df_.dropna(subset=['DTSM'])
        except:
            print('\tNo DTSM column, no na dropped!')
            
        if strict_input_output and (len(target_mnemonics) != len(df_.columns)):
            print(f'\tNo all target mnemonics are found in df, strict_input_output rule applied, returned None!')
            return None
        elif not strict_input_output and (len(target_mnemonics) != len(df_.columns)):
            print(f'\tNo all target mnemonics are in df, returned PARTIAL dataframe!')
            return df_.dropna(axis=0)
        else:
            print(f'\tAll target mnemonics are found in df, returned COMPLETE dataframe!')
            return df_.dropna(axis=0)


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