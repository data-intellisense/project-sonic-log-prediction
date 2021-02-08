#%% import lib
import pathlib
import pickle
import re

import lasio
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from scipy.signal import medfilt
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator, RegressorMixin

# given a mnemonic, find all of its alias
def get_alias(mnemonic, alias_dict=None):
    assert alias_dict is not None, 'alias_dict is None, assign alias_dict!'
    alias_dict = alias_dict or {}
    return [k for k, v in alias_dict.items() if mnemonic in v]


# given a alias, find its corresponding one and only mnemonic
# return a mnemonis if found, or else ''
def get_mnemonic(alias=None, alias_dict=None):
    assert alias_dict is not None, 'alias_dict is None, assign alias_dict!'
    alias_dict = alias_dict or {}
    try:
        return [v for k, v in alias_dict.items() if alias == k][0]
    except:
        return ""


#%% read las, return curves and data etc.


class read_las:
    def __init__(self, las):
        self.las = lasio.read(las)
        return None

    def df(self):
        return self.las.df()

    def df_curvedata(self):
        l = []
        for i in self.las.curves:
            l.append([i.mnemonic, i.unit, i.descr])
        return pd.DataFrame(l, columns=["mnemonic", "unit", "description"])

    def get_mnemonic_unit(self):
        return self.df_curvedata()[["mnemonic", "unit"]]

    def get_curvedata(self, data_names=[]):
        df = self.df_curves()

    def df_welldata(self, valid_value_only=True):
        l = []
        for i in self.las.well:
            if valid_value_only and (i.value != ""):
                l.append([i.mnemonic, i.unit, i.value, i.descr])
        return pd.DataFrame(l, columns=["mnemonic", "unit", "value", "description"])

    def get_welldata(self, data_names=[]):
        df = self.df_welldata()
        return [df[df["mnemonic"] == i]["value"].values[0] for i in data_names]

    def get_lat_lon(self, data_names=["SLAT", "SLON"]):
        return self.get_welldata(data_names=data_names)

    def get_start_stop(self, data_names=["STRT", "STOP"]):
        return self.get_welldata(data_names=data_names)


#% TEST: read_las()
# convert las.curves info to df for better reading

if __name__ == "__main__":
    las = "data/las/0052442d0162_TGS.las"
    las = read_las(las)
    df = las.df()
    print(df)
    print(las.df_curvedata())
    print(las.df_welldata())
    print(las.get_mnemonic_unit())
    print("lat and lon:", las.get_lat_lon())
    print("start-stop depth:", las.get_start_stop())


#%%
def sample_weight_calc(length=1, decay=0.999):

    assert all([decay > 0, decay <= 1])
    assert all([length >= 1, type(length) is int])
    return decay ** np.arange(length, 0, step=-1)


def get_distance(a, b):
    """
    a and b are lists: [lat, lon] of two locations
    """
    assert isinstance(a, list)
    assert isinstance(b, list)
    assert all([len(a) == 2, len(b) == 2])

    return np.sum(np.square(np.array(a) - np.array(b))) ** 0.5


def get_sample_weight(las_name=None, las_dict=None, las_lat_lon=None):
    """
    sample weight based on horizontal distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight = []
    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            sample_weight = sample_weight + [
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k])
            ] * len(las_dict[k])
    sample_weight = np.array(sample_weight).reshape(-1, 1)
    sample_weight = MinMaxScaler().fit_transform(sample_weight)

    return sample_weight


def get_distance_weight(las_name=None, las_dict=None, las_lat_lon=None):
    """
    sample weight based on horizontal distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    distance_weight = dict()

    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            distance_weight[k] = (
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k]) * 10
            )
            if distance_weight[k] > 10:
                distance_weight[k] = 10

    distance_weight[las_name] = max(distance_weight.values()) * 2

    return distance_weight


def get_sample_weight2(
    las_name=None,
    las_dict=None,
    vertical_anisotropy=0.01,
    las_lat_lon=None,
):
    """
    sample weight based on horizontal and vertical distance between wells
    """
    assert (las_name, str)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight1 = []
    sample_weight2 = []
    for k in sorted(las_dict.keys()):
        if k not in [las_name]:
            sample_weight1 = sample_weight1 + [
                1 / get_distance(las_lat_lon[las_name], las_lat_lon[k])
            ] * len(las_dict[k])

            avg = np.mean(las_dict[las_name].index)
            vertical_distance = list(abs(np.array(las_dict[k].index) - avg))
            sample_weight2 = sample_weight2 + vertical_distance

    sample_weight1 = np.array(sample_weight1).reshape(-1, 1)
    sample_weight1 = MinMaxScaler().fit_transform(sample_weight1)

    sample_weight2 = np.array(sample_weight2).reshape(-1, 1)
    sample_weight2 = MinMaxScaler().fit_transform(sample_weight2)

    sample_weight = np.sqrt(
        sample_weight1 ** 2 + (1 / vertical_anisotropy * sample_weight2) ** 2
    )

    return sample_weight


def get_sample_weight2_TEST(
    lat_lon_TEST=None,
    mid_depth_TEST=None,
    las_dict=None,
    vertical_anisotropy=0.01,
    las_lat_lon=None,
):
    """
    sample weight based on horizontal and vertical distance between wells
    """
    assert (lat_lon_TEST, list)
    assert (las_dict, dict)

    # get sample weight = 1/distance between target las and remaining las
    sample_weight1 = []
    sample_weight2 = []
    for k in sorted(las_dict.keys()):

        sample_weight1 = sample_weight1 + [
            1 / get_distance(lat_lon_TEST, las_lat_lon[k])
        ] * len(las_dict[k])

        vertical_distance = list(abs(np.array(las_dict[k].index) - mid_depth_TEST))
        sample_weight2 = sample_weight2 + vertical_distance

    sample_weight1 = np.array(sample_weight1).reshape(-1, 1)
    sample_weight1 = MinMaxScaler().fit_transform(sample_weight1)

    sample_weight2 = np.array(sample_weight2).reshape(-1, 1)
    sample_weight2 = MinMaxScaler().fit_transform(sample_weight2)

    sample_weight = np.sqrt(
        sample_weight1 ** 2 + (1 / vertical_anisotropy * sample_weight2) ** 2
    )

    return sample_weight


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


#%% process las data


class process_las:
    def __init__(self):
        return None

    def _renumber_columns(self, c):
        c = list(c.values)
        assert isinstance(c, list)
        c = c.copy()
        for i in c:
            if c.count(i) > 1:
                ix = [j for j, val in enumerate(c) if val == i]
                for m, n in enumerate(ix):
                    c[n] = c[n] + "_" + str(m)
        return c

    def despike(self, df=None, cols=None, window_size=5):
        """
        df should be a dataframe, will despike all or selected columns
        """
        assert isinstance(df, pd.DataFrame)
        assert window_size % 2 == 1, "Median filter window size must be odd."

        if cols is not None:
            assert isinstance(cols, list), "cols should be a list columns names of df"
            assert all(
                [i in df.columns for i in cols]
            ), "cols should be a list columns names of df"
        else:
            cols = df.columns

        for col in cols:
            df[col] = medfilt(df[col].values, kernel_size=window_size)

        return df

    def keep_valid_DTSM_only(self, df=None):
        """
        return a df with rows of valid DTSM data
        """
        
        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        
        # make sure only one 'DTSM' exist
        if list(df.columns.values).count("DTSM") > 1:
            print("More than one 'DTSM' curve exist!")

        if "DTSM" not in df.columns:
            print("DTSM not present in this las file!")

        # drop all empty rows in 'DTSM' column if there is a 'DTSM' column
        # if 'DTSM' in df.columns, dropp all rows with nan in DTSM column
        df = df.dropna(subset=["DTSM"])

        # drop other columns with all na
        df = df.dropna(axis=1, how="all")

        return df

    def load_test_No_DTSM(self, df=None, alias_dict=None):
        """
        return a df with selected mnemonics/alias
        """

        assert alias_dict is not None, 'alias_dict is None, assign alias_dict!'

        # deep copy of df so manipulation here won't alter original df
        df = df.copy()

        # drop other columns with all na
        df = df.dropna(axis=1, how="all")

        return df

    def get_df_by_mnemonics(
        self, df=None, target_mnemonics=None, alias_dict=None, strict_input_output=True, drop_na=True, log_RT=True,
    ):
        """
        useage: get a cleaned dataframe by given mnemonics,
        target_mnemonics: a list of legal mnemonics,
        strict_input_output: if true, only output a df only if all target mnemonics are found in las, output None otherwise
            if false, will output a df with all possible mnemonis found in las
        """
        assert alias_dict is not None, 'alias_dict is None, assign alias_dict!'
        df = df.copy()
        # check required parameters
        if target_mnemonics is None:
            print(
                "Target mnemonics are required as 'get_df_by_mnemonics' function input!"
            )
            return None
        else:
            assert isinstance(target_mnemonics, list)
            assert len(target_mnemonics) >= 1
            assert all(
                [i in alias_dict.values() for i in target_mnemonics]
            ), f"Mnemonics should be in the list of {np.unique(list(alias_dict.values()))}"

        # find alias in df for target mnemonics
        alias = dict()
        for m in target_mnemonics:
            alias[m] = []
            for col in df.columns:
                n = get_mnemonic(col, alias_dict=alias_dict)
                if m == n:
                    alias[m].append(col)

        df_ = None
        df_cols = []

        # key: target mnenomics; value: alias in df, average the curves if more than 1 exist!
        for key, value in alias.items():
            if len(value) == 1:
                temp = df[value].values.reshape(-1, 1)
                df_cols.append(key)
            elif len(value) > 1:
                temp = df[value].mean(axis=1).values.reshape(-1, 1)
                df_cols.append(key)
                # print(f'{key} has more than 1 curves:{value}, averaged!')
            elif len(value) == 0:  # return index if no such column
                # print(f'\tNo corresponding alias for {key}!')
                continue

            if df_ is None:
                df_ = temp
            else:
                df_ = np.c_[df_, temp]
        if df_ is None:
            return None
        else:
            df_ = pd.DataFrame(df_, columns=df_cols)
            df_.index = df.index

        # add a 'DEPTH' feature as it may be useful
        # df_['DEPTH'] = df.index

        # dropped rows with na in DTSM column
        # try:
        #     df_ = df_.dropna(subset=['DTSM'])
        # except:
        #     print('\tNo DTSM column, no na dropped!')

        if drop_na:
            df_ = df_.dropna(axis=0)
        
        if 'RT' in df_.columns:
            df_['RT'] = abs(df_['RT'])+1
            if log_RT:
                df_['RT'] = np.log(df_['RT'])

        if strict_input_output and (len(target_mnemonics) != len(df_.columns)):
            print(
                f"\tNo all target mnemonics are found in df, strict_input_output rule applied, returned None!"
            )
            return None

        elif not strict_input_output and (len(target_mnemonics) != len(df_.columns)):
            print(f"\tNo all target mnemonics are in df, returned PARTIAL dataframe!")
            # better to drop all na in all columns
            return df_

        else:
            print(
                f"\tAll target mnemonics are found in df, returned COMPLETE dataframe!"
            )
            # better to drop all na in all columns
            return df_

    def get_compiled_df_from_las_dict(
        self,
        las_data_dict=None,
        target_mnemonics=None,
        alias_dict=None,
        strict_input_output=True,
        add_DEPTH_col=True,
        log_RT=True,
        return_dict=False,
        ):
            
        target_las_dict = dict()
        # get the data that corresponds to terget mnemonics
        for key in las_data_dict.keys():
            print(f"Loading {key}")
            df = las_data_dict[key]

            df = self.despike(df, window_size=5)

            df = self.get_df_by_mnemonics(
                df=df, target_mnemonics=target_mnemonics, strict_input_output=True, alias_dict=alias_dict,
                log_RT=log_RT
            )

            if (df is not None) and len(df > 1):
                
                # add 'DEPTH' as a feature and rearrange columns
                if add_DEPTH_col:
                    df['DEPTH']=df.index 
                    cols = df.columns.to_list()
                    df = df[cols[-1:]+cols[:-1]]

                target_las_dict[key] = df
        
        print(
            f"Total {len(target_las_dict.keys())} las files loaded and total {sum([len(i) for i in target_las_dict.values()])} rows of data!"
        )

        df_ = pd.concat([target_las_dict[k] for k in target_las_dict.keys()], axis=0)                                    

        if return_dict:
            return target_las_dict
        else:
            return df_


class MeanRegressor(BaseEstimator, RegressorMixin):
    def __init__(self):
        return None

    def fit(self, X_train=None, y_train=None):
        # the prediction will always be the mean of y
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.DataFrame)
        assert len(y_train)==len(X_train)
        assert y_train.shape[1]==1

        self.y_bar_ = np.mean(y_train.values)
        self.y = y_train
        return self.y_bar_

    def predict(self, X_test):
        # give back the mean of y, in the same length as input X
        return np.ones(len(X_test)) * self.y_bar_

#%% Test data
las_name_test = "001-00a60e5cc262_TGS"
las_test = "data/las/00a60e5cc262_TGS.las"
df_test = read_las(las_test).df()
