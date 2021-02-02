#%% this module is to used test different models
import os
import pathlib
import pickle
import random
import time
import re
import glob

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

from plot import plot_crossplot, plot_logs_columns

# from models.nn_model import nn_model
# load customized functions and requried dataset
from util import (
    CV_weighted,
    alias_dict,
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    las_data_DTSM_QC,
    process_las,
)

from util import las_lat_lon as las_lat_lon_TRAIN

pio.renderers.default = "browser"

# change working directory to current file directory
# path = r"C:\Users\Julian Liu\Documents\Project\SPEGCS ML Challenge\project-gcs-datathon2021"
path = pathlib.Path(__file__).parent
os.chdir(path)


# get las_lat_lon
with open(f"{path}/data/leaderboard_1/las_lat_lon.pickle", "rb") as f:
    las_lat_lon_TEST = pickle.load(f)

# get las_data_TEST
with open(f"{path}/data/leaderboard_1/las_data_TEST.pickle", "rb") as f:
    las_data_TEST = pickle.load(f)

#%%  TEST 2: split train/test among las files (recommended)


def test_predict(
    target_mnemonics=None,
    model=None,
    df_TEST=None,
    las_data_DTSM=None,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder=None,
):
    TEST_folder = f"{path}/predictions/{TEST_folder}"
    if not os.path.exists(TEST_folder):
        os.mkdir(TEST_folder)

    # Do NOT include 'DTSM'!!
    target_mnemonics_TEST = target_mnemonics
    target_mnemonics_TRAIN = target_mnemonics + ["DTSM"]

    time0 = time.time()

    # prepare TEST data with terget mnemonics
    df_TEST = df_TEST.copy()

    if despike:
        df_TEST = process_las().despike(df_TEST, window_size=5)

    print(df_TEST)
    df_TEST = process_las().get_df_by_mnemonics(
        df=df_TEST,
        target_mnemonics=target_mnemonics_TEST,
        strict_input_output=True,
        drop_na=False,
    )

    print(df_TEST)
    print("Data df_test shape:", df_TEST.shape)
    print("Selected df_test columns:", df_TEST.columns)

    X_test = df_TEST.values

    # prepare TRAIN data with terget mnemonics
    las_dict = dict()
    for key in las_data_DTSM.keys():
        print(f"Loading {key}")
        df = las_data_DTSM[key]

        if despike:
            df = process_las().despike(df, window_size=5)

        df = process_las().get_df_by_mnemonics(
            df=df,
            target_mnemonics=target_mnemonics_TRAIN,
            strict_input_output=True,
            drop_na=True,
        )

        if (df is not None) and len(df > 1):
            las_dict[key] = df
    print(
        f"Total {len(las_dict.keys())} las files loaded and total {sum([len(i) for i in las_dict.values()])} rows of data!"
    )

    # train and predict on TEST data

    # create training data dataframe
    Xy_train = pd.concat([las_dict[k] for k in las_dict.keys()], axis=0)

    X_train = Xy_train.values[:, :-1]
    y_train = Xy_train.values[:, -1:]

    # scale train data
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # get sample weight for training
    # if sample_weight_type == 2:
    #     sample_weight = get_sample_weight2(
    #         las_name=las_name,
    #         las_lat_lon=las_lat_lon,
    #         las_data_DTSM=las_data_DTSM_QC,
    #         las_dict=las_dict,
    #         vertical_anisotropy=0.01,
    #     )
    # 0 or any other value will lead to no sample weight used
    # else:
    #     sample_weight = None

    sample_weight = None
    # fit the model
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)

    # scale test data and predict, and scale back prediction
    X_test = scaler_x.transform(X_test)
    y_predict = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))
    y_predict = pd.DataFrame(y_predict, columns=["Predicted DTSM"])

    y_predict.index = df_TEST.index
    # if len(df_TEST) == len(y_predict):
    #     y_predict.index = df_TEST.index
    # else:
    #     y_predict = pd.merge(df_TEST, y)

    print(
        f"Completed traing and predicting on TEST data in time {time.time()-time0:.2f} seconds"
    )

    return y_predict


#%%  Predict on TEST data, Group1, 7features

Group1 = [
    "002-Well_02",
    "003-Well_03",
    "007-Well_07",
    "008-Well_08",
    "009-Well_09",
]

# choose the best model with tuned hyper parameters
model_7 = XGB(
    tree_method="hist",
    objective="reg:squarederror",
    subsample=0.76,
    n_estimators=250,
    min_child_weight=0.02,
    max_depth=3,
    learning_rate=0.052,
    reg_lambda=33,
)

#%  choose 7 features/predictors (not including 'DTSM')
TEST_folder = "TEST"
target_mnemonics = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]

# WellName = "002-Well_02"
# df_TEST = las_data_TEST[WellName]

#% check mnemonics in TEST data
# process_las().get_df_by_mnemonics(
#     df=df_TEST, target_mnemonics=target_mnemonics, strict_input_output=True
# )

for WellName in Group1:

    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics,
        model=model_7,
        df_TEST=df_TEST,
        las_data_DTSM=las_data_DTSM_QC,
        las_lat_lon_TEST=None,
        las_lat_lon=None,
        sample_weight_type=2,
        despike=True,
        TEST_folder="TEST",
    )

    y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")
    print("X_test and y_predict length:", len(df_TEST), len(y_predict))
    print(f"Prediction results are saved at: predictions/TEST")

#%% check predicted data: "001-Well_01"

#% top part with 7 features
target_mnemonics_7 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
model_7 = XGB(
    tree_method="hist",
    objective="reg:squarederror",
    subsample=0.76,
    n_estimators=250,
    min_child_weight=0.02,
    max_depth=3,
    learning_rate=0.052,
    reg_lambda=33,
)


WellName = "001-Well_01"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


# rename NPHI
df_TEST = df_TEST.rename(columns={"SPHI_LS": "NPHI"})
df_TEST_7 = df_TEST[df_TEST.index <= 7900]
df_TEST_7.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_7,
    model=model_7,
    df_TEST=df_TEST_7,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_7features.csv")

#% model with 3 features: ["DTCO", "NPHI", "GR"], version 1
#% bottom part with 3 features
target_mnemonics_3 = ["DTCO", "NPHI", "GR"]

params = {
    "subsample": 0.9999999999999999,
    "n_estimators": 200,
    "min_child_weight": 0.23,
    "max_depth": 5,
    "learning_rate": 0.029470517025518096,
    "lambda": 68,
}
model_3_1 = XGB(**params)
df_TEST_3 = df_TEST  # [df_TEST.index > 7900]
df_TEST_3.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_3,
    model=model_3_1,
    df_TEST=df_TEST_3,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_3features.csv")

print(f"Prediction results are saved at: predictions/TEST")


#%% check predicted data: "004-Well_04"

#% top part with 6 features
# target_mnemonics_7 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT"]

params = {
    "subsample": 0.7999999999999999,
    "n_estimators": 100,
    "min_child_weight": 0.39,
    "max_depth": 5,
    "learning_rate": 0.0625055192527397,
    "lambda": 59,
}
model_6 = XGB(**params)

WellName = "004-Well_04"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


y_predict = test_predict(
    target_mnemonics=target_mnemonics_6,
    model=model_6,
    df_TEST=df_TEST,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_6features.csv")

#%% check predicted data: "005-Well_05"

#% top part with 6 features
target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT"]

params = {
    "subsample": 0.7999999999999999,
    "n_estimators": 100,
    "min_child_weight": 0.39,
    "max_depth": 5,
    "learning_rate": 0.0625055192527397,
    "lambda": 59,
}
model_6 = XGB(**params)

WellName = "005-Well_05"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


y_predict = test_predict(
    target_mnemonics=target_mnemonics_6,
    model=model_6,
    df_TEST=df_TEST,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")


#%% check predicted data: "006-Well_06"

#% top part with 6 features
target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "PEFZ"]

params = {
    "subsample": 0.7,
    "n_estimators": 200,
    "min_child_weight": 0.03,
    "max_depth": 4,
    "learning_rate": 0.03556480306223128,
    "lambda": 36,
}
model_6 = XGB(**params)

WellName = "006-Well_06"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))

y_predict = test_predict(
    target_mnemonics=target_mnemonics_6,
    model=model_6,
    df_TEST=df_TEST,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")


#%% check predicted data: "010-Well_10"

#% top part with 7 features
target_mnemonics_7 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
model_7 = XGB(
    tree_method="hist",
    objective="reg:squarederror",
    subsample=0.76,
    n_estimators=250,
    min_child_weight=0.02,
    max_depth=3,
    learning_rate=0.052,
    reg_lambda=33,
)


WellName = "010-Well_10"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


# rename NPHI
df_TEST_7 = df_TEST[df_TEST.index >= 8700]
df_TEST_7.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_7,
    model=model_7,
    df_TEST=df_TEST_7,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_7features.csv")

#% model with 3 features: ["DTCO", "NPHI", "GR"], version 1
#% bottom part with 3 features
target_mnemonics_3 = ["DTCO", "GR", "RT"]

params = {
    "subsample": 0.7999999999999999,
    "n_estimators": 250,
    "min_child_weight": 0.22,
    "max_depth": 5,
    "learning_rate": 0.03906939937054615,
    "lambda": 18,
}
model_3_1 = XGB(**params)
df_TEST_3 = df_TEST
df_TEST_3.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_3,
    model=model_3_1,
    df_TEST=df_TEST_3,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    despike=True,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_3features.csv")

print(f"Prediction results are saved at: predictions/TEST")


#%% check df length

for WellName in las_data_TEST.keys():
    print(WellName, las_data_TEST[WellName].shape)

for f in glob.glob("predictions/TEST/*.csv"):
    f_name = re.split("[\\\/]", f)[-1][-15:-4]

    df_ypred = pd.read_csv(f)
    df_ypred.columns = ["Depth", "DTSM_Pred"]

    plot_logs_columns(
        df=las_data_TEST[f_name],
        DTSM_pred=df_ypred,
        well_name=f_name,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"XGB-{f_name}-Prediction-Depth",
        plot_save_path=f"{path}/predictions/TEST/plots",
        plot_save_format=["png", "html"],  # availabe format: ["png", "html"]
    )
