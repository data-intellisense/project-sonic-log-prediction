
import pickle

#%% mnemonics dictionary
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
    print(get_alias('DPHI', alias_dict=alias_dict))
    print(get_mnemonic(alias = 'GR', alias_dict=alias_dict))

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
            
#%% get_features_df


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