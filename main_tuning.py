#%% this module is to used test different models
import os
import pathlib
import pickle
import random
import time

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.ensemble import GradientBoostingRegressor as GBR
from sklearn.ensemble import StackingRegressor as Stack

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor as XGB

from sklearn.model_selection import RandomizedSearchCV


from plot import plot_crossplot, plot_logs_columns

# from models.nn_model import nn_model
# load customized functions and requried dataset
from util import (CV_weighted, alias_dict, get_alias, get_mnemonic,
                  get_sample_weight, get_sample_weight2, las_data_DTSM,
                  process_las)

pio.renderers.default='browser'

# change working directory to current file directory
path = pathlib.Path(__file__).parent
os.chdir(path)
# path = r"C:\Users\Julian Liu\Documents\Project\SPEGCS ML Challenge\project-gcs-datathon2021"

#%%  TEST 2: split train/test among las files (recommended)

#%%  TEST 2: split train/test among las files (recommended)

# choose 7 features/predictors (not including 'DTSM')
TEST_folder = '7features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'DPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']

# folder to store plots, will create one if not exists
TEST_folder = '2features_LOOCV_las'
target_mnemonics = ['DTCO', 'RHOB']

# folder to store plots, will create one if not exists
TEST_folder = '5features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'RT']

# folder to store plots, will create one if not exists
TEST_folder = '3features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'RHOB']

# choose 8 features/predictors (not including 'DTSM')
TEST_folder = '8features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'DPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']


if not os.path.exists(f'{path}/predictions/{TEST_folder}'):
    os.mkdir(f'{path}/predictions/{TEST_folder}')

target_mnemonics = target_mnemonics + ['DTSM'] # 'DTSM' is a response variable
las_dict = dict()

# get the data that corresponds to terget mnemonics
for key in las_data_DTSM.keys():
    print(f'Loading {key}')
    df = las_data_DTSM[key]

    df = process_las().despike(df, window_size=5)
    
    df = process_las().get_df_by_mnemonics(df=df, target_mnemonics=target_mnemonics, strict_input_output=True)

    if (df is not None) and len(df>1):
        las_dict[key] = df

print(f'Total {len(las_dict.keys())} las files loaded and total {sum([len(i) for i in las_dict.values()])} rows of data!')

Xy = pd.concat([las_dict[k] for k in las_dict.keys() ], axis=0)

X_train = Xy.iloc[:, :-1]
y_train = Xy.iloc[:, -1:]

#%%
param_distributions = {
    "n_estimators": range(100, 300, 50),
    "max_depth": range(3, 12),
    "min_child_weight": np.arange(0.01, 0.4, 0.01),
    "learning_rate": np.logspace(-3, -1),
    "subsample": np.arange(0.7, 1.0, 0.1),
    "lambda": range(1, 100)
}


RandCV = RandomizedSearchCV(
            estimator=XGB(tree_method='hist', objective='reg:squarederror'),
            param_distributions=param_distributions,
            n_iter=10,
            scoring='neg_mean_squared_error',
            verbose=2,
)

RandCV.fit(X=X_train, y=y_train)

print('best parameters:', tune_search.best_params_) 
print('Completed training with all models!')


