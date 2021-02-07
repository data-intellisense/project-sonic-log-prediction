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
import xgboost

from sklearn.model_selection import RandomizedSearchCV

from bayes_opt import BayesianOptimization

from plot import plot_crossplot, plot_logs_columns

from load_pickle import alias_dict, las_data_DTSM_QC
# load customized functions and requried dataset
from util import (    
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,    
    process_las,
)

pio.renderers.default = "browser"

# change working directory to current file directory
path = pathlib.Path(__file__).parent
os.chdir(path)
# path = r"C:\Users\Julian Liu\Documents\Project\SPEGCS ML Challenge\project-gcs-datathon2021"

#%%  TEST 2: split train/test among las files (recommended)

#%%  TEST 2: split train/test among las files (recommended)

# # choose 7 features/predictors (not including 'DTSM')
TEST_folder = "7features_LOOCV_las"
target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

# TEST_folder = "6features_LOOCV_las"
# target_mnemonics = ["RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

# # choose 3 features/predictors (not including 'DTSM')
# TEST_folder = "3features_LOOCV_las"
# target_mnemonics = ["DTCO", "NPHI", "GR"]

# # choose 3 features/predictors (not including 'DTSM')
# TEST_folder = "3features_LOOCV_las2"
# target_mnemonics = ["DTCO", "GR", "RT"]

# # # choose 6 features/predictors (not including 'DTSM')
# TEST_folder = "6features_LOOCV_las"
# target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI"]

# # # choose 6 features/predictors (not including 'DTSM')
# TEST_folder = "6features_LOOCV_las2"
# target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ"]


if not os.path.exists(f"{path}/predictions/{TEST_folder}"):
    os.mkdir(f"{path}/predictions/{TEST_folder}")

target_mnemonics = target_mnemonics + ["DTSM"]  # 'DTSM' is a response variable
las_dict = dict()

# get the data that corresponds to terget mnemonics
for key in las_data_DTSM_QC.keys():
    print(f"Loading {key}")
    df = las_data_DTSM_QC[key]

    df = process_las().despike(df, window_size=5)

    df = process_las().get_df_by_mnemonics(
        df=df, target_mnemonics=target_mnemonics, strict_input_output=True, alias_dict=alias_dict
    )

    if (df is not None) and len(df > 1):
        
        # add 'DEPTH' as a feature and rearrange columns
        df['DEPTH']=df.index 
        cols = df.columns.to_list()
        df = df[cols[-1:]+cols[:-1]]

        las_dict[key] = df

print(
    f"Total {len(las_dict.keys())} las files loaded and total {sum([len(i) for i in las_dict.values()])} rows of data!"
)

Xy = pd.concat([las_dict[k] for k in las_dict.keys()], axis=0)

X_train = Xy.iloc[:, :-1]
y_train = Xy.iloc[:, -1:]

dtrain = xgboost.DMatrix(data=X_train, label=y_train)

#%% create XGB data matrix 
time0=time.time()

params_7 = {'subsample': 0.7999999999999999, 
            'min_child_weight': 0.31, 'max_depth': 3, 
            'learning_rate': 0.022229964825261943, 'lambda': 96}
cv_results = xgboost.cv(params_7, dtrain=dtrain, num_boost_round=10, nfold=3,
    early_stopping_rounds=10, metrics='rmse')
print(cv_results.iloc[-1,-2])
print(f'finished in {time.time()-time0: .2f} s')


#%% XGB tuning

# param_distributions = {
#     "n_estimators": range(100, 300, 50),
#     "max_depth": range(3, 12),
#     "min_child_weight": np.arange(0.01, 0.4, 0.01),
#     "learning_rate": np.logspace(-3, -1),
#     "subsample": np.arange(0.7, 1.0, 0.1),
#     "lambda": range(1, 100),
# }


# RandCV = RandomizedSearchCV(
#     estimator=XGB(tree_method="hist", objective="reg:squarederror"),
#     param_distributions=param_distributions,
#     n_iter=10,
#     scoring="neg_mean_squared_error",
#     CV=5,
#     verbose=2,
# )

# RandCV.fit(X=X_train, y=y_train)

# RandCV.cv_results_
# print("best parameters:", RandCV.best_params_)
# print("Completed training with all models!")

# target_mnemonics = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
params_7 = {'subsample': 0.7999999999999999, 'n_estimators': 250, 
            'min_child_weight': 0.31, 'max_depth': 3, 
            'learning_rate': 0.022229964825261943, 'lambda': 96}
model_xgb =XGB(**params_7, tree_method="hist", objective="reg:squarederror")

model_xgb.fit(X=X_train, y=y_train)

xgb_feature_importance = [round(i,3) for i in model_xgb.feature_importances_]
print(f'Feature importance:\n{df.columns} \n{xgb_feature_importance}')

xgb_feature_importance_df = pd.DataFrame(np.c_[df.columns[:-1].values.reshape(-1,1), 
    np.array(xgb_feature_importance).reshape(-1,1)], columns=['feature', 'importance'])
px.bar(xgb_feature_importance_df, x='feature', y='importance')


#%% XGB Bayesian optimization

time0=time.time()

def xgb_func(**params):
    # params["num_boost_round"]=int(params["num_boost_round"])
    params['max_depth']=int(params['max_depth'])
    cv_results = xgboost.cv(params, dtrain=dtrain, nfold=5, num_boost_round=30,
        early_stopping_rounds=10, metrics='rmse')
    return(-cv_results.iloc[-1,-2])

param_distributions = {
    "max_depth": (3, 12),
    "min_child_weight": (0.01, 0.5),
    "learning_rate": (0.0001, 0.1),
    "subsample": (0.7, 1.0),
    "lambda": (1, 100),
}


# initialize the BayesOpt
xgb_bayes = BayesianOptimization(f=xgb_func,
    pbounds=param_distributions,
    verbose=3,    
    random_state=42)

xgb_bayes.maximize(
    n_iter=30,
    init_points=3,
)
print(xgb_bayes.max)
print(f'finished in {time.time()-time0: .2f} s')


#%% XGB hyperopt

time0=time.time()

def xgb_func(params):

    cv_results = xgboost.cv(params, dtrain=dtrain, nfold=5, num_boost_round=30,
        early_stopping_rounds=10, metrics='rmse')
    return {'loss': (cv_results.iloc[-1,-2]), "status": STATUS_OK}

from hyperopt import tpe, hp, fmin, STATUS_OK,Trials

space = {
    # "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.choice("max_depth", range(1,15)),
    "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.5),
    "learning_rate": hp.loguniform("learning_rate", -4, -1),
    "subsample": hp.uniform("subsample", 0.7, 1.0),
    # "lambda": hp.loguniform("lambda", 0, 2),

}

trials = Trials()
best = fmin(
    fn=xgb_func, 
    space=space,
    algo=tpe.suggest,
    max_evals=10,
    trials=trials
)

print(f'Best:{best}')
# best rmse loss: 6.0976406
# Best:{'learning_rate': 0.17921801478908123, 'max_depth': 13, 'min_child_weight': 0.4571378438073825, 'subsample': 0.9917229046738573}
#%% MLP tuning

param_distributions = {
    "hidden_layer_sizes": [(100,), (30, 30), (200,), (20, 20, 20)],
    "alpha": [0.00001, 0.0001, 0.001, 0.01],  
    'learning_rate_init'  : [0.001, 0.01, 0.1]
}

model = MLP(random_state=42, learning_rate='adaptive', activation='relu', max_iter=200, early_stopping=True)

RandCV = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="neg_mean_squared_error",
    verbose=2,
)

RandCV.fit(X=X_train, y=y_train)

print("best parameters:", RandCV.best_params_)
print("Completed training with all models!")

# {'learning_rate_init': 0.01, 'hidden_layer_sizes': (100,), 'alpha': 0.001}