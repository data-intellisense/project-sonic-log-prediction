#%% this module is to used test different models
import os
import pathlib
import pickle
import random
import time
import glob
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import RobustScaler, StandardScaler
from xgboost import XGBRegressor as XGB
from sklearn.base import clone

from plot import plot_crossplot, plot_logs_columns

SelectedScaler = RobustScaler

from load_pickle import (
    las_data_DTSM_QC,
    las_lat_lon,
    alias_dict,
    las_depth,
    las_lat_lon,
    test_list,
)

# load customized functions and requried dataset
from util import (
    to_pkl,
    read_pkl,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    get_nearest_neighbors,
)

pio.renderers.default = "browser"

#%% LOOCV


def LOOCV_evaluate(
    target_mnemonics=None,
    models=None,
    scaling=True,
    scalers=None,
    path=None,
    las_data_DTSM=None,
    las_lat_lon=None,
    sample_weight_type=None,
):

    if not os.path.exists(path):
        os.mkdir(path)

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    # setup recording rmse_las for each model_dict
    rmse_all_las = []
    rmse_all_las_temp = []

    # evaluate the model with the pre-defined 107 las files
    for _, las_name in test_list.itertuples():
        Xy_test = las_data_DTSM_QC[las_name].copy()
        Xy_test = process_las().get_df_by_mnemonics(
            df=Xy_test,
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=True,
            alias_dict=alias_dict,
            outliers_contamination=None,  # should not remove outliers when testing!!!
            drop_na=True,  # drop_na should be False when predicting 20 test files
        )

        # reset rmse_las for each model_dict
        rmse_las = []
        y_predict_models = []

        for model_name, model in models.items():

            # print("model:", model)
            time0 = time.time()

            # scale test data and predict, and scale back prediction
            X_test = Xy_test.values[:, :-1]
            y_test = Xy_test.values[:, -1:]

            # print(X_test.shape, X_test)
            if scaling:
                X_test = scalers[0].transform(X_test)
                y_predict = scalers[1].inverse_transform(
                    model.predict(X_test).reshape(-1, 1)
                )
            else:
                y_predict = model.predict(X_test).reshape(-1, 1)

            y_predict_models.append(y_predict)

        y_predict_models = np.stack(y_predict_models, axis=1)
        y_predict = np.mean(y_predict_models, axis=1)

        # calculate rmse_las
        rmse_las = mean_squared_error(y_test, y_predict) ** 0.5

        print(f"{las_name} rmse: {rmse_las:.2f} \trun in {time.time()-time0:.1f} s")

        # plot crossplot to compare y_predict vs y_actual
        plot_crossplot(
            y_actual=y_test,
            y_predict=y_predict,
            include_diagnal_line=True,
            text=None,
            plot_show=False,
            plot_return=False,
            plot_save_file_name=f"{model_name}-{las_name}-Prediction-Crossplot",
            plot_save_path=path,
            plot_save_format=["png"],  # availabe format: ["png", "html"]
        )

        # plot pred vs true DTSM
        # df_ypred with proper column names as pd.DataFrame is required for proper plotting
        df_ypred = pd.DataFrame(
            np.c_[Xy_test.index.values.reshape(-1, 1), y_predict.reshape(-1, 1)],
            columns=["Depth", "DTSM_Pred"],
        )
        plot_logs_columns(
            df=Xy_test,
            DTSM_pred=df_ypred,
            well_name=las_name,
            alias_dict=alias_dict,
            plot_show=False,
            plot_return=False,
            plot_save_file_name=f"{model_name}-{las_name}-Prediction-Depth",
            plot_save_path=path,
            plot_save_format=["png"],
        )

        # saving rmse for each las prediction
        rmse_all_las_temp.append(rmse_las)
        rmse_all_las.append([las_name, len(df_ypred), rmse_las])
        rmse_mean = np.mean(rmse_all_las_temp)
        print(f"{model_name} model_dict with mean rmse so far: {rmse_mean:.2f}")

    rmse_all_las = pd.DataFrame(rmse_all_las, columns=["las_name", "rows", "rmse"])
    rmse_all_las.to_csv(f"{path}/rmse_all_las_{model_name}.csv")

    rmse_corrected = (
        sum((rmse_all_las["rows"] * rmse_all_las["rmse"] ** 2))
        / sum(rmse_all_las["rows"])
    ) ** 0.5
    print(f"{model_name} model_dict with corrected rmse : {rmse_corrected:.2f}")

    return [rmse_mean, rmse_corrected]


#%%  LOOCV_evaluate
path = f"predictions/evaluate/6_2_despike"

if not os.path.exists(path):
    os.mkdir(path)

for model_path in glob.glob(f"{path}/*.pickle"):

    # load the model_dict
    model_dict = read_pkl(model_path)

    print(
        model_dict["model_name"], model_dict["target_mnemonics"], model_dict["rmse_CV"]
    )
    print("\n")

    model_name = model_dict["model_name"]
    models = {f"XGB_{model_name}": model_dict["best_estimator"]}

    # from models.models import models
    time0 = time.time()

    rmse_LOOCV = LOOCV_evaluate(
        target_mnemonics=model_dict["target_mnemonics"],
        models=models,
        scaling=True,
        scalers=[model_dict["scaler_x"], model_dict["scaler_y"]],
        path=path,
        las_data_DTSM=las_data_DTSM_QC,
        las_lat_lon=las_lat_lon,
        sample_weight_type=None,
    )

    # update the model_dict with rmse_LOOCV and save it!
    model_dict["rmse_LOOCV_mean"] = rmse_LOOCV[0]
    model_dict["rmse_LOOCV_corr"] = rmse_LOOCV[1]
    to_pkl(model_dict, f"{path}/model_xgb_{model_name}.pickle")

    print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")
    print(f"Prediction results are saved at: {path}")

# checkout the save model
for model in glob.glob(f"{path}/*.pickle"):
    print(read_pkl(model))
#%% check out the models

path = "models/GroupKFoldCV"

temp = []

for model in glob.glob(f"{path}/*.pickle"):

    # load the model_dict
    model_dict = read_pkl(model)
    keys = [
        "model_name",
        "rmse_LOOCV",
        "rmse_LOOCV_mean",
        "rmse_LOOCV_corr",
        "rmse_CV",
        "target_mnemonics",
    ]
    keys_ = [key for key in keys if key in model_dict.keys()]
    temp.append([model_dict[key] for key in keys_])

# create a dataframe for sorting
temp = pd.DataFrame(temp, columns=keys_)

for col in ["rmse_LOOCV_corr", "rmse_LOOCV_mean", "rmse_LOOCV", "rmse_CV"]:

    if col in temp.columns:
        print(temp.sort_values(by=[col], ascending=True))
