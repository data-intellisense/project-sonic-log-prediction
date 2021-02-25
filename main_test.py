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
    las_data_TEST_renamed,
)


# load customized functions and requried dataset
from util import (
    read_pkl,
    # get_alias,
    # get_mnemonic,
)

pio.renderers.default = "browser"


#%%  TEST 2: split train/test among las files (recommended)


def test_predict(
    target_mnemonics=None,
    model=None,
    scalers=None,
    df_TEST=None,
    las_data_DTSM=None,
    lat_lon_TEST=None,
    las_lat_lon=None,
    sample_weight_type=None,
    TEST_folder=None,
):

    # start counting time
    time0 = time.time()
    if not os.path.exists(TEST_folder):
        os.mkdir(TEST_folder)

    # Do NOT include 'DTSM'!!
    target_mnemonics_TEST = [i for i in target_mnemonics if i != "DTSM"]
    print(
        "\ntarget_mnemonics:",
        target_mnemonics,
        "\ntarget_mnemonics_TEST:",
        target_mnemonics_TEST,
    )

    # prepare TEST data with terget mnemonics
    df_TEST = df_TEST.copy()

    # df_TEST = process_las().despike(df_TEST, window_size=5)
    df_TEST = process_las().get_df_by_mnemonics(
        df=df_TEST,
        target_mnemonics=target_mnemonics_TEST,
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        drop_na=False,
    )

    # make sure the column sequence is the same as target_mnemonics
    # print("df_TEST", df_TEST.head())
    X_test = df_TEST.values

    # scale test data and predict, and scale back prediction
    X_test = scalers[0].transform(X_test)
    y_predict = scalers[1].inverse_transform(model.predict(X_test).reshape(-1, 1))
    y_predict = pd.DataFrame(y_predict, columns=["Pred DTSM"])

    y_predict.index = df_TEST.index
    print(
        f"Completed traing and predicting on TEST data in time {time.time()-time0:.2f} seconds"
    )

    return y_predict


#% load existing models
models_TEST = dict()
for path in glob.glob("models/models_TEST/*.pickle"):
    # load the model_dict
    model_dict = read_pkl(path)
    print(model_dict["model_name"], model_dict["target_mnemonics"])

    models_TEST[model_dict["model_name"]] = model_dict

TEST_folder = "predictions/TEST"

#%% Well 2,3,6,7,8,9 using model 5_1, Well 4,5 use model 5_2

model_names = ["5_1", "5_1", "5_1", "5_1", "5_1", "5_1", "5_2", "5_2"]
Group = [
    "002-Well_02",
    "003-Well_03",
    "006-Well_06",  # have to use 6_2
    "007-Well_07",
    "008-Well_08",
    "009-Well_09",
    "004-Well_04",
    "005-Well_05",
]


for ix, WellName in enumerate(Group):

    model_name = model_names[ix]
    target_mnemonics = models_TEST[model_name]["target_mnemonics"]
    model = models_TEST[model_name]["best_estimator"]
    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics,
        model=model,
        scalers=[
            models_TEST[model_name]["scaler_x"],
            models_TEST[model_name]["scaler_y"],
        ],
        df_TEST=df_TEST,
        TEST_folder=TEST_folder,
    )

    y_predict.to_csv(f"{TEST_folder}/Prediction.{WellName}.csv")
    print(f"Predictions for {WellName} are saved at: {TEST_folder}")

#%% Well 01
model_names = ["5_1", "3_1"]
Group = ["001-Well_01", "001-Well_01"]

y_predict_ = []
for ix, WellName in enumerate(Group):

    model_name = model_names[ix]
    target_mnemonics = models_TEST[model_name]["target_mnemonics"]
    model = models_TEST[model_name]["best_estimator"]
    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics,
        model=model,
        scalers=[
            models_TEST[model_name]["scaler_x"],
            models_TEST[model_name]["scaler_y"],
        ],
        df_TEST=df_TEST,
        TEST_folder=TEST_folder,
    )
    y_predict_.append(y_predict)


# combined prediction
y_predict = pd.concat(
    [
        y_predict_[0][y_predict_[0].index <= 7900],
        y_predict_[1][y_predict_[1].index > 7900],
    ],
    axis=0,
)

y_predict.to_csv(f"{TEST_folder}/Prediction.{WellName}.csv")
print(f"Predictions for {WellName} are saved at: {TEST_folder}")

#%% Well 10
model_names = ["3_2", "5_1"]
Group = ["010-Well_10", "010-Well_10"]

y_predict_ = []
for ix, WellName in enumerate(Group):

    model_name = model_names[ix]
    target_mnemonics = models_TEST[model_name]["target_mnemonics"]
    model = models_TEST[model_name]["best_estimator"]
    df_TEST = las_data_TEST[WellName]

    y_predict = test_predict(
        target_mnemonics=target_mnemonics,
        model=model,
        scalers=[
            models_TEST[model_name]["scaler_x"],
            models_TEST[model_name]["scaler_y"],
        ],
        df_TEST=df_TEST,
        TEST_folder=TEST_folder,
    )
    y_predict_.append(y_predict)

# combined prediction
y_predict = pd.concat(
    [
        y_predict_[0][y_predict_[0].index <= 8650],
        y_predict_[1][y_predict_[1].index > 8650],
    ],
    axis=0,
)

y_predict.to_csv(f"{TEST_folder}/Prediction.{WellName}.csv")
print(f"Predictions for {WellName} are saved at: {TEST_folder}")


#%% check df length, and plot the prediction and do a visual check on prediction quality

for f in glob.glob(f"{TEST_folder}/*.csv"):
    f_name = re.split("[\\\/.]", f)[-2]

    df_ypred = pd.read_csv(f)

    # rename columns for plotting
    df_ypred.columns = ["Depth", "DTSM_Pred"]

    # make sure the predicted depth is the exactly the same as raw depth
    all(df_ypred.index == las_data_TEST[f_name].index)
    print(
        f_name,
        "\nraw data shape\t\t",
        las_data_TEST[f_name].shape,
        "\npredict data shape\t",
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
    df_ypred.columns = ["Depth", "DTSM"]
    f_name_ = int(re.split("[_]", f_name)[-1])
    df_ypred.to_excel(f"{TEST_folder}/Well {f_name_}.xlsx", index=False)
