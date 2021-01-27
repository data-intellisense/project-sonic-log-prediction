#%% import lib
import numpy as np
import pickle
# for CV_weighted
from sklearn.model_selection import KFold
from sklearn.base import clone
import lasio 
import pandas as pd 
import re
# for metrics
from sklearn.metrics import mean_squared_error
import pathlib

#%% mnemonics dictionary
path = pathlib.Path(__file__).parent
#os.chdir(path)  # change current working directory, os.getcwd()

# get the alias_dict, required
with open(f'{path}/data/alias_dict.pickle', 'rb') as f:
    alias_dict = pickle.load(f)

# given a mnemonic, find all of its alias
def get_alias(mnemonic, alias_dict=None):
    alias_dict = alias_dict or {}
    return [k for k, v in alias_dict.items() if mnemonic in v]

# given a alias, find its corresponding one and only mnemonic
# return a mnemonis if found, or else ''
def get_mnemonic(alias = None, alias_dict=None):
    alias_dict = alias_dict or {}
    try:
        return [v for k, v in alias_dict.items() if alias == k][0]
    except:
        return ''

if __name__ == '__main__':    

    print(get_alias('PEFZ', alias_dict=alias_dict))

    print(get_mnemonic(alias = 'MODT', alias_dict=alias_dict))

def sample_weight_calc(length=1, decay=0.999):

    assert all([decay > 0, decay <= 1])
    assert all([length >= 1, type(length) is int])
    return decay ** np.arange(length, 0, step=-1)

def CV_weighted(model, X, y, weights=None, cv=10):
    """
    model : a sci-kit learn estimator
    X : a numpy array of shape (n_samples, n_features)
    y : numpy array of shape (n_samples,)
    weights : sample weights
    cv : TYPE, optional, The default is 10.
    metrics : TYPE, optional, The default is [mean_squared_error].
    Returns: scores
    """

    if weights is None:
        weights = np.ones(len(X))

    kf = KFold(n_splits=cv)
    kf.get_n_splits(X)
    scores = []
    for train_index, test_index in kf.split(X):
        model_clone = clone(model)
                
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        weights_train, weights_test = weights[train_index], weights[test_index]

        try:
            model_clone.fit(X_train, y_train, sample_weight=weights_train)
        except: 
            # KNN, MLP does not accept sample_weight
            model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_test)

        score = mean_squared_error(y_test, y_pred, sample_weight=weights_test)
        
        scores.append(score)

    return np.mean(scores)
#%% read las, return curves and data etc.

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

#%% TEST: read_las() from util.py 
# convert las.curves info to df for better reading

if __name__ == '__main__':    
    las = "data/las/0052442d0162_TGS.las"
    las = read_las(las)
    print(las.df())
    print(las.df_curvedata())
    print(las.df_welldata())
    print(las.get_mnemonic_unit())
    print('lat and lon:', las.get_lat_lon())
    print('start-stop depth:', las.get_start_stop())

#%% process las data

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

    def keep_valid_DTSM_only(self, df=None, alias_dict=alias_dict):
        '''
        keep rows with valid DTSM data
        '''    
        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        # keep only column names
        # columns_old = df.columns.copy()

        # convert all alias to desired mnemonics
        # df.columns = df.columns.map(alias_dict)                

        # drop all columns whose mnemonics are not recognized, i.e. NaN
        # df = df.loc[:,df.columns.notnull()]
        #df.columns = df.columns.fillna('to_drop')

        # attach suffix number to duplicate column names
        # df.columns = self.renumber_columns(df.columns)

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
        useage: get a cleaned dataframe by given mnemonics
        target_mnemonics: a list of legal mnemonics
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
                print(f'\tNo corresponding alias for {key}!')
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
            print(f'\tNo all target mnemonics are in df, strict_input_output rule applied, return None!')
            return None
        elif not strict_input_output and (len(target_mnemonics) != len(df_.columns)):
            print(f'\tNo all target mnemonics are in df, returned PARTIAL dataframe!')
            return df_.dropna(axis=0)
        else:
            print(f'\tAll target mnemonics are found in df, returned COMPLETE dataframe!')
            return df_.dropna(axis=0)


#%% TEST get_df_by_mnemonics

if __name__ == '__main__':    
    las = "data/las/0052442d0162_TGS.las"
    df = read_las(las).df()
    
    print('original df:', df.head(5))
    print('\nnew df:', process_las().get_df_by_mnemonics(df=df, target_mnemonics=['DTCO', 'GR', 'DTSM'], strict_input_output=False))

    print('\nnew df:', process_las().get_df_by_mnemonics(df=df, target_mnemonics=['DTCO', 'GR', 'DPHI', 'DTSM'], strict_input_output=False))

    print('\nnew df:', process_las().get_df_by_mnemonics(df=df, target_mnemonics=['DTCO', 'GR', 'DPHI', 'DTSM'], strict_input_output=True))