#%% import lib
import numpy as np
import pickle
# for CV_weighted
from sklearn.model_selection import KFold
from sklearn.base import clone
import lasio 
import pandas as pd 

# for metrics
from sklearn.metrics import mean_squared_error


#%% mnemonics dictionary

# get the alias_dict, required
with open('data/alias_dict.pickle', 'rb') as f:
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

    print(get_alias('DTSM', alias_dict=alias_dict))

    print(get_mnemonic(alias = 'MODT', alias_dict=alias_dict))

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

        # keep only column names
        # columns_old = df.columns.copy()

        # convert all alias to desired mnemonics
        df.columns = df.columns.map(alias_dict)                

        # drop all columns whose mnemonics are not recognized, i.e. NaN
        df = df.loc[:,df.columns.notnull()]
        #df.columns = df.columns.fillna('to_drop')

        # attach suffix number to duplicate column names
        # df.columns = self.renumber_columns(df.columns)

        # make sure only one 'DTSM' exist
        try:
            list(df.columns.values).count('DTSM') <= 1
        except:
            print("More than one 'DTSM' curve exist!")        

        if 'DTSM' not in df.columns:
            print('DTSM not in this las file!')

        # drop all empty rows in 'DTSM' column if there is a 'DTSM' column
        # if 'DTSM' in df.columns, dropp all rows with nan in DTSM column
        df = df.dropna(subset=['DTSM'])

        # drop other columns with all na
        df = df.dropna(axis=1, how='all')

        return df  
        
def get_features_df(df, features=None):

    assert isinstance(features, list), 'features must be a list of curve mnemonics'

    # if no features provided, then return all possible/useful features
    if features is None:
        features = ['DTSM', 'DTCO', 'DPHI', 'NPHI', 'RHOB', 'GR', 'AT', 'RT']

    col_features = []
    for col in df.columns:
        if get_mnemonic(col) in features:
            col_features.append(col)

    if 'DTSM' not in col_features:
        print('No DTSM in df for get_features_df!')

    return df[col_features]

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


#%% get_mnemonics_4df

def get_df_from_mnemonics(df, target_mnemonics=None):
    alias = dict()
    for m in target_mnemonics:
        alias[m] = []
        for col in df.columns:
            n = get_mnemonic(col, alias_dict=alias_dict)
            if (m==n):            
                alias[m].append(col)

    df_ = None
    df_cols = []
    for key, value in alias.items():
        if len(value)==1:
            temp = df[value].values.reshape(-1,1)
        elif len(value)>1:
            temp = df[value].mean(axis=1).values.reshape(-1,1)
        elif len(value)==0: # return index if no such column
            temp = df.index.values.reshape(-1,1)

        if df_ is None:
            df_ = temp
            df_cols = [key]
        else:
            df_ = np.c_[df_, temp]
            df_cols = df_cols + [key]

    df_ = pd.DataFrame(df_, columns=df_cols)
    df_.index = df.index
    df_ = df_.dropna(subset=['DTSM'])

    return df_
