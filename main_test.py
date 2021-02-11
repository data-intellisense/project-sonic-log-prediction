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

from plot import plot_crossplot, plot_logs_columns

from load_pickle import (
    alias_dict,
    las_data_DTSM_QC,
    las_lat_lon,
    lat_lon_TEST,    
    las_data_TEST,
)

# models for prediction
from models.models import (
    model_xgb_7,
    model_xgb_3_1,
    model_xgb_3_2,
    model_xgb_6_1,
    model_xgb_6_2,
)

# load customized functions and requried dataset
from util import (
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    get_sample_weight2_TEST,
)

pio.renderers.default = "browser"


#%%  TEST 2: split train/test among las files (recommended)


def test_predict(
    target_mnemonics=None,
    model=None,
    df_TEST=None,
    las_data_DTSM=None,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    TEST_folder=None,
):
    TEST_folder = f"predictions/{TEST_folder}"
    if not os.path.exists(TEST_folder):
        os.mkdir(TEST_folder)

    # Do NOT include 'DTSM'!!
    target_mnemonics_TEST = target_mnemonics
    target_mnemonics_TRAIN = target_mnemonics + ["DTSM"]

    time0 = time.time()

    # prepare TEST data with terget mnemonics
    df_TEST = df_TEST.copy()

    df_TEST = process_las().despike(df_TEST, window_size=5)

    print("df_TEST", df_TEST)
    df_TEST = process_las().get_df_by_mnemonics(
        df=df_TEST,
        target_mnemonics=target_mnemonics_TEST,
        strict_input_output=True,
        alias_dict=alias_dict,
        add_DEPTH_col=True,
        drop_na=False,
    )

    print("Data df_test shape:", df_TEST.shape)
    print("Selected df_test columns:", df_TEST.columns)

    X_test = df_TEST.values

    # prepare TRAIN data with terget mnemonics
    las_dict = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC,
        target_mnemonics=target_mnemonics_TRAIN,
        alias_dict=alias_dict,
        strict_input_output=True,
        add_DEPTH_col=True,
        log_RT=True,
        return_dict=True,
    )

    Xy_train = pd.concat([las_dict[k] for k in las_dict.keys()], axis=0)

    X_train = Xy_train.iloc[:, :-1]
    y_train = Xy_train.iloc[:, -1:]

    # scale train data
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    # get sample weight for training
    if sample_weight_type == 2:
        sample_weight = get_sample_weight2_TEST(
            lat_lon_TEST=lat_lon_TEST,
            mid_depth_TEST=df_TEST.index.values.mean(),
            las_dict=las_dict,
            vertical_anisotropy=0.01,
            las_lat_lon=las_lat_lon,
        )
    # 0 or any other value will lead to no sample weight used
    else:
        sample_weight = None

    # fit the model
    if sample_weight is not None:
        model.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
        print(
            "Model does not accept sample weight so sample weight was not used in training!"
        )

    # scale test data and predict, and scale back prediction
    X_test = scaler_x.transform(X_test)
    y_predict = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))
    y_predict = pd.DataFrame(y_predict, columns=["Predicted DTSM"])

    y_predict.index = df_TEST.index
    print(
        f"Completed traing and predicting on TEST data in time {time.time()-time0:.2f} seconds"
    )

    return y_predict


#%%  Predict on TEST data, Group1, 7features

# folder to store TEST prediction results
TEST_folder = "TEST"

Group1 = [
    "002-Well_02",
    "003-Well_03",
    "007-Well_07",
    "008-Well_08",
    "009-Well_09",
]

#%  choose 7 features/predictors (not including 'DTSM')
target_mnemonics_7 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]

for WellName in Group1:

    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics_7,
        model=model_xgb_7,
        df_TEST=df_TEST,
        las_data_DTSM=las_data_DTSM_QC,
        lat_lon_TEST=lat_lon_TEST[WellName],
        las_lat_lon=las_lat_lon,
        sample_weight_type=2,
        TEST_folder="TEST",
    )

    y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")
    print("X_test and y_predict length:", len(df_TEST), len(y_predict))
    print(f"Prediction results are saved at: predictions/TEST")

#%% check predicted data: "001-Well_01"

#% top part with 7 features

WellName = "001-Well_01"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


df_TEST_7 = df_TEST[df_TEST.index <= 7900]
df_TEST_7.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_7,
    model=model_7,
    df_TEST=df_TEST_7,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_7features.csv")

#% bottom part with 3 features, model_3_1
target_mnemonics_3_1 = ["DTCO", "NPHI", "GR"]

df_TEST_3 = df_TEST[df_TEST.index > 7900]
df_TEST_3.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_3_1,
    model=model_xgb_3_1,
    df_TEST=df_TEST_3,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_3features.csv")

print(f"Prediction results are saved at: predictions/TEST")


#%% check predicted data: ["004-Well_04", "005-Well_05"]

#%  choose 7 features/predictors (not including 'DTSM')
target_mnemonics_6_1 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT"]

Group2 = ["004-Well_04", "005-Well_05"]

for WellName in Group2:

    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics_6_1,
        model=model_xgb_6_1,
        df_TEST=df_TEST,
        las_data_DTSM=las_data_DTSM_QC,
        lat_lon_TEST=lat_lon_TEST[WellName],
        las_lat_lon=las_lat_lon,
        sample_weight_type=2,
        TEST_folder="TEST",
    )

    y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")
    print("X_test and y_predict length:", len(df_TEST), len(y_predict))
    print(f"Prediction results are saved at: predictions/TEST")


#%% check predicted data: "006-Well_06"

#% top part with 6 features
target_mnemonics_6_2 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "PEFZ"]

WellName = "006-Well_06"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))

y_predict = test_predict(
    target_mnemonics=target_mnemonics_6_2,
    model=model_xgb_6_2,
    df_TEST=df_TEST,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")


#%% check predicted data: "010-Well_10"

#% top part with 7 features
target_mnemonics_7 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]

WellName = "010-Well_10"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))


# rename NPHI
df_TEST_7 = df_TEST[df_TEST.index >= 8700]
df_TEST_7.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_7,
    model=model_xgb_7,
    df_TEST=df_TEST_7,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
    TEST_folder="TEST",
)

y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}_7features.csv")

#% model with 3 features: ["DTCO", "NPHI", "GR"], version 1
#% bottom part with 3 features
target_mnemonics_3_2 = ["DTCO", "GR", "RT"]

df_TEST_3 = df_TEST[df_TEST.index < 8700]
df_TEST_3.shape

y_predict = test_predict(
    target_mnemonics=target_mnemonics_3_2,
    model=model_3_1,
    df_TEST=df_TEST_3,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=2,
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
        plot_save_path=f"predictions/TEST/plots",
        plot_save_format=["png", "html"],  # availabe format: ["png", "html"]
    )
