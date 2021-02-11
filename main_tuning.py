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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from pprint import pprint
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor as XGB
import xgboost

from sklearn.model_selection import RandomizedSearchCV

# from bayes_opt import BayesianOptimization

from plot import plot_crossplot, plot_logs_columns

from load_pickle import alias_dict, las_data_DTSM_QC


# load customized functions and requried dataset
from util import (
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    MeanRegressor,
    to_pkl,
    read_pkl,
)

pio.renderers.default = "browser"

# change working directory to current file directory if necessary
# path = pathlib.Path(__file__).parent
# # os.chdir(path)

# #%% Baseline model, y_mean and linear regression

# models = {"Mean-Model": MeanRegressor(), "MLR": LinearRegression(), "RCV": RidgeCV()}

# for model_name, model in models.items():
#     model.fit(X_train, y_train)

#     scores = cross_val_score(
#         model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"
#     )
#     print(f"{model_name} rmse:\t{-np.mean(scores):.2f}")

# # Mean: 0.6
#%%  TEST 2: split train/test among las files (recommended)

# # # choose 7 features/predictors (not including 'DTSM')
# TEST_folder_7 = "7features_LOOCV_las"
# target_mnemonics_7 = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

# # choose 3 features/predictors (not including 'DTSM')
# TEST_folder_3_1 = "3features_LOOCV_las"
# target_mnemonics_3_1 = ["DTCO", "NPHI", "RT"]

# TEST_folder_2 = "3features_LOOCV_las"
# target_mnemonics_2 = ["DTCO", "NPHI"]

# TEST_folder = TEST_folder_7
# target_mnemonics = target_mnemonics_7

# if not os.path.exists(f"predictions/{TEST_folder}"):
#     os.mkdir(f"predictions/{TEST_folder}")

# target_mnemonics = target_mnemonics + ["DTSM"]

# Xy = process_las().get_compiled_df_from_las_dict(
#         las_data_dict=las_data_DTSM_QC,
#         target_mnemonics=target_mnemonics,
#         new_mnemonics=['DEPTH'],
#         log_mnemonics=["RT"],
#         strict_input_output=True,
#         alias_dict=alias_dict,
#         return_dict=False,
#         drop_na=True,
#     )

# print(Xy.sample(10))

# scaler_x, scaler_y = RobustScaler(), RobustScaler()
# X_train = scaler_x.fit_transform(Xy.iloc[:, :-1])
# y_train = scaler_y.fit_transform( Xy.iloc[:, -1:])

# #%% Single RandomizedSearchCV to tune XGB

# param_distributions = {
#     "n_estimators": range(100, 300, 20),
#     "max_depth": range(1, 9),
#     "min_child_weight": np.arange(0.01, 0.4, 0.01),
#     "learning_rate": np.logspace(-3, -1),
#     "subsample": np.arange(0.8, 1.02, 0.02),
#     "colsample_bytree": np.arange(0.8, 1.02, 0.02),
#     "lambda": range(1, 100),
#     "gamma": range(0,10),
# }

# time0 = time.time()
# RandCV = RandomizedSearchCV(
#     estimator=XGB(tree_method="hist", objective="reg:squarederror"),
#     param_distributions=param_distributions,
#     n_iter=30,
#     scoring="neg_root_mean_squared_error",
#     cv=5,
#     verbose=2,
# )

# RandCV.fit(X=X_train, y=y_train)

# XGB_7_RandCV_results = RandCV.cv_results_

# pprint(XGB_7_RandCV_results)
# print("best parameters:", RandCV.best_params_)
# XGB_7_RandCV_results['best_params_'] = RandCV.best_params_
# to_pkl(XGB_7_RandCV_results, 'predictions/tuning/XGB_7_RandCV_results.pickle' )

# print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")


# # params_7 = read_pkl('predictions/tuning/XGB_7_RandCV_results.pickle')['best_params_']

# params_7 = {'subsample': 0.94,
#  'n_estimators': 140,
#  'min_child_weight': 0.09,
#  'max_depth': 3,
#  'learning_rate': 0.03906939937054615,
#  'gamma': 5,
#  'colsample_bytree': 0.94}

# model_xgb = XGB(**params_7, tree_method="hist", objective="reg:squarederror")

# model_xgb.fit(X=X_train, y=y_train)

# xgb_feature_importance = [round(i, 3) for i in model_xgb.feature_importances_]
# print(f"Feature importance:\n{Xy.columns} \n{xgb_feature_importance}")

# xgb_feature_importance_df = pd.DataFrame(
#     np.c_[
#         Xy.columns[:-1].values.reshape(-1, 1),
#         np.array(xgb_feature_importance).reshape(-1, 1),
#     ],
#     columns=["feature", "importance"],
# )

# fig = px.bar(xgb_feature_importance_df, x="feature", y="importance", width=1600, height=900)
# fig.show()
# fig.write_image("readme_resources/xgb_7feature_importance.png")

# y_predict = model_xgb.predict(X_train).reshape(-1,1)

# plot_crossplot(
#     y_actual=scaler_y.inverse_transform(y_train).reshape(-1,1),
#     y_predict=scaler_y.inverse_transform(y_predict).reshape(-1,1),
#     text=None,
#     axis_range=300,
#     include_diagnal_line=True,
#     plot_show=True,
#     plot_return=False,
#     plot_save_file_name='XGB tuning cross plot',
#     plot_save_path='predictions/tuning',
#     plot_save_format= ["png", "html"],
# )

#%% Batch tuning XGB using RandomizedSearchCV

mnemonic_dict = {
    # "DTSM" as response
    # '7':   ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    # '7_1': ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    # '7_2': ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    # '6_1': ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    # '6_2': ["DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    # '3_1': ["DTCO", "NPHI", "GR", "DTSM"],
    # '3_2': ["DTCO",  "GR", "RT", "DTSM"],
    # "DTCO" as response,for well 6 and 8, to fix DTCO
    "DTCO_5": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTCO"],
    "DTCO_6": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "RT", "DTCO"],
}

# create a dictionary to save all the models
model_xgb = dict()

time0 = time.time()

for model_name, target_mnemonics in mnemonic_dict.items():

    Xy = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC,
        target_mnemonics=target_mnemonics,
        new_mnemonics=["DEPTH"],
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        return_dict=False,
        drop_na=True,
    )

    # print(Xy.sample(10))

    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(Xy.iloc[:, :-1])
    y_train = scaler_y.fit_transform(Xy.iloc[:, -1:])

    # RandomizedSearchCV to find best hyperparameter combination

    param_distributions = {
        "n_estimators": range(100, 300, 20),
        "max_depth": range(1, 9),
        "min_child_weight": np.arange(0.01, 0.4, 0.01),
        "learning_rate": np.logspace(-3, -1),
        "subsample": np.arange(0.8, 1.02, 0.02),
        "colsample_bytree": np.arange(0.8, 1.02, 0.02),
        "gamma": range(0, 10),
    }

    RandCV = RandomizedSearchCV(
        estimator=XGB(tree_method="gpu_hist", objective="reg:squarederror"),
        param_distributions=param_distributions,
        n_iter=100,
        scoring="neg_root_mean_squared_error",
        cv=5,
        verbose=2,
    )

    RandCV.fit(X=X_train, y=y_train)

    model_xgb[f"model_xgb_{model_name}"] = XGB(
        **RandCV.best_params_, tree_method="gpu_hist", objective="reg:squarederror"
    )
    # print("best parameters:", RandCV.best_params_)

    # retrain the model with best parameters
    model_xgb[f"model_xgb_{model_name}"].fit(X=X_train, y=y_train)

    # save all models to pickle file each iteration, in case of crashes
    to_pkl(model_xgb, f"predictions/tuning/Tuned_Trained_XGB_Models.pickle")

print(f"\nCompleted training and saved all models in {time.time()-time0:.1f} seconds!")

# models=read_pkl(f'predictions/tuning/Tuned_Trained_XGB_Models.pickle' )
# print(models)