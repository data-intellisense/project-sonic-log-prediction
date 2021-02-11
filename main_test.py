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

from load_pickle import (
    alias_dict,
    las_data_DTSM_QC,
    las_lat_lon,
    lat_lon_TEST,
    las_data_TEST,
)


# load customized functions and requried dataset
from util import (
    read_pkl,
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
    sample_weight_type=None,
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

    # df_TEST = process_las().despike(df_TEST, window_size=5)

    # print("df_TEST", df_TEST.head())
    # print(target_mnemonics_TEST)

    df_TEST = process_las().get_df_by_mnemonics(
        df=df_TEST,
        target_mnemonics=target_mnemonics_TEST,
        new_mnemonics=["DEPTH"],
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        drop_na=False,
    )

    # print("Data df_test shape:", df_TEST.shape)
    # print("Selected df_test columns:", df_TEST.columns)

    # make sure the column sequence is the same as target_mnemonics
    df_TEST = df_TEST[target_mnemonics_TEST]
    X_test = df_TEST.values

    # just need to get the scaler
    Xy = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC,
        target_mnemonics=target_mnemonics_TRAIN,
        new_mnemonics=["DEPTH"],
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        drop_na=True,
        return_dict=False,
    )

    Xy = Xy[target_mnemonics_TRAIN]
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(Xy.values[:, :-1])
    y_train = scaler_y.fit_transform(Xy.values[:, -1:])

    model.fit(X_train, y_train)

    assert all(
        i in Xy.columns[:-1] for i in df_TEST.columns
    ), "Train and test data should have the same column except 'DTSM'!"

    # print(Xy.head())
    # print(df_TEST.head())

    # scale test data and predict, and scale back prediction
    X_test = scaler_x.transform(X_test)
    y_predict = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))
    y_predict = pd.DataFrame(y_predict, columns=["Predicted DTSM"])

    y_predict.index = df_TEST.index
    print(
        f"Completed traing and predicting on TEST data in time {time.time()-time0:.2f} seconds"
    )

    return y_predict


#%% load existing models

mnemonic_dict = {
    # "DTSM" as response
    "7": ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    "7_1": ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    "7_2": ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    "6_1": ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    "6_2": ["DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    "3_1": ["DTCO", "NPHI", "GR", "DTSM"],
    "3_2": ["DTCO", "GR", "RT", "DTSM"],
    # "DTCO" as response,for well 6 and 8, to fix DTCO
    # "DTCO_5": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTCO"],
    # "DTCO_6": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "RT", "DTCO"],
}


# load the model
model = read_pkl("models/Tuned_Trained_XGB_Models_DTSM.pickle")

model_xgb_6_2 = model["model_xgb_6_2"]
model_xgb_6_1 = model["model_xgb_6_1"]
model_xgb_3_1 = model["model_xgb_3_1"]
model_xgb_3_2 = model["model_xgb_3_2"]


#%%  Predict on TEST data, Group1, 7features

# folder to store TEST prediction results
TEST_folder = "TEST"

Group1 = [
    "002-Well_02",
    "003-Well_03",
    "006-Well_06",  # have to use 6_2
    "007-Well_07",
    "008-Well_08",
    "009-Well_09",
]

#%  choose 7 features/predictors (not including 'DTSM')
target_mnemonics_6_2 = mnemonic_dict["6_2"][:-1]

# temp replace 6_2 model
# target_mnemonics_6_2 = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]
# params_xgb = {
#     "subsample": 0.8999999999999999,
#     "n_estimators": 150,
#     "min_child_weight": 0.08,
#     "max_depth": 3,
#     "learning_rate": 0.03906939937054615,
#     "lambda": 31,
# }
# model_xgb_6_2 = XGB(**params_xgb, tree_method="gpu_hist", objective="reg:squarederror")

for WellName in Group1:

    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics_6_2,
        model=model_xgb_6_2,
        df_TEST=df_TEST,
        las_data_DTSM=las_data_DTSM_QC,
        lat_lon_TEST=None,
        las_lat_lon=None,
        sample_weight_type=None,
        TEST_folder="TEST",
    )

    y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")
    print("X_test and y_predict length:", len(df_TEST), len(y_predict))
    print(f"Prediction results are saved at: predictions/{TEST_folder}")

#%% check predicted data: "001-Well_01"

WellName = "001-Well_01"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))

#% top part with 7 features
df_TEST_01A = df_TEST[df_TEST.index <= 7900]
df_TEST_01A.shape

y_predict_01A = test_predict(
    target_mnemonics=target_mnemonics_6_2,
    model=model_xgb_6_2,
    df_TEST=df_TEST_01A,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=None,
    TEST_folder="TEST",
)

#% bottom part with 3 features, model_3_1
target_mnemonics_3_1 = mnemonic_dict["3_1"][:-1]

df_TEST_01B = df_TEST[df_TEST.index > 7900]
df_TEST_01B.shape

y_predict_01B = test_predict(
    target_mnemonics=target_mnemonics_3_1,
    model=model_xgb_3_1,
    df_TEST=df_TEST_01B,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=None,
    TEST_folder="TEST",
)

y_predict = pd.concat([y_predict_01A, y_predict_01B])
y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")

print(f"Prediction results are saved at: predictions/TEST")


#%% check predicted data: ["004-Well_04", "005-Well_05"]

#%  choose 7 features/predictors (not including 'DTSM')
target_mnemonics_6_1 = mnemonic_dict["6_1"][:-1]

Group2 = ["004-Well_04", "005-Well_05"]

for WellName in Group2:

    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics_6_1,
        model=model_xgb_6_1,
        df_TEST=df_TEST,
        las_data_DTSM=las_data_DTSM_QC,
        lat_lon_TEST=None,
        las_lat_lon=las_lat_lon,
        sample_weight_type=None,
        TEST_folder="TEST",
    )

    y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")
    print("X_test and y_predict length:", len(df_TEST), len(y_predict))
    print(f"Prediction results are saved at: predictions/TEST")


#%% check predicted data: "010-Well_10"

WellName = "010-Well_10"
df_TEST = las_data_TEST[WellName]
print("Total row of data:", len(df_TEST))

#% top part with 3 features
target_mnemonics_3_2 = mnemonic_dict["3_2"][:-1]

df_TEST_10A = df_TEST[df_TEST.index < 8700]
df_TEST_10A.shape

y_predict_10A = test_predict(
    target_mnemonics=target_mnemonics_3_2,
    model=model_xgb_3_2,
    df_TEST=df_TEST_10A,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=None,
    TEST_folder="TEST",
)

# bottom part
df_TEST_10B = df_TEST[df_TEST.index >= 8700]
df_TEST_10B.shape

y_predict_10B = test_predict(
    target_mnemonics=target_mnemonics_6_2,
    model=model_xgb_6_2,
    df_TEST=df_TEST_10B,
    las_data_DTSM=las_data_DTSM_QC,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=None,
    TEST_folder="TEST",
)

y_predict = pd.concat([y_predict_10A, y_predict_10B])
y_predict.to_csv(f"predictions/TEST/Prediction_{WellName}.csv")

print(f"Prediction results are saved at: predictions/TEST")

# assert 1 == 2, "make sure to update the test well, fix missing data and fix DTCO"
#%% check df length, and plot the prediction and do a visual check on prediction quality


for f in glob.glob("predictions/TEST/*.csv"):
    f_name = re.split("[\\\/]", f)[-1][-15:-4]

    df_ypred = pd.read_csv(f)
    df_ypred.columns = ["Depth", "DTSM_Pred"]

    print(
        f_name,
        "raw data shape",
        las_data_TEST[f_name].shape,
        "predict data shape",
        df_ypred.shape,
    )
    plot_logs_columns(
        df=las_data_TEST[f_name],
        DTSM_pred=df_ypred,
        well_name=f_name,
        alias_dict=alias_dict,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"XGB-{f_name}-Prediction-Depth",
        plot_save_path=f"predictions/TEST/plots",
        plot_save_format=["png", "html"],  # availabe format: ["png", "html"]
    )


#%% convert to submission format

for f in glob.glob("predictions/TEST/*.csv"):
    f_name = re.split("[\\\/]", f)[-1][-11:-4]

    df_ypred = pd.read_csv(f)
    df_ypred.columns = ["Depth", "DTSM"]

    df_ypred.to_excel(f"predictions/TEST/to_submit/{f_name}.xlsx", index=False)
