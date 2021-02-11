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
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
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
)

# load customized functions and requried dataset
from util import (
    to_pkl,
    read_pkl,
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    get_nearest_neighbors,
)

pio.renderers.default = "browser"

# change working directory to current file directory
# path = pathlib.Path(__file__).parent
# os.chdir(path)


#%%  TEST 2: split train/test among las files (recommended)


def LOOCV(
    target_mnemonics=None,
    models=None,
    scaling=True,
    TEST_folder=None,
    las_data_DTSM=None,
    las_lat_lon=None,
    sample_weight_type=None,
):

    if not os.path.exists(f"predictions/{TEST_folder}"):
        os.mkdir(f"predictions/{TEST_folder}")

    target_mnemonics = target_mnemonics + ["DTSM"]

    las_dict = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC,
        target_mnemonics=target_mnemonics,
        alias_dict=alias_dict,
        strict_input_output=True,
        add_DEPTH_col=True,
        log_mnemonics=["RT"],
        return_dict=True,
    )

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    # setup recording rmse_las for each model
    rmse_all_las = []
    rmse_all_las_temp = []

    # create test/train data
    for las_name in las_dict.keys():

        # reset rmse_las for each model
        rmse_las = []
        y_predict_models = []

        # # get nearest neighbors for training
        # neighbors = get_nearest_neighbors(
        #     depth_TEST=las_depth[las_name],
        #     lat_lon_TEST=las_lat_lon[las_name],
        #     las_depth=las_depth,
        #     las_lat_lon=las_lat_lon,
        #     num_of_neighbors=20,
        #     vertical_anisotropy=1,
        #     depth_range_weight=0.1,
        # )
        # las_dict = dict()

        # for k in neighbors:
        #     las_dict[k[0]] = las_dict_all_las[k[0]]

        for model_name, model in models.items():

            time0 = time.time()

            # use one las file as test data
            Xy_test = las_dict[las_name]
            Xy_train = pd.concat(
                [las_dict[k] for k in las_dict.keys() if k != las_name], axis=0
            )

            X_train = Xy_train.values[:, :-1]
            y_train = Xy_train.values[:, -1:]

            # scale train data
            if scaling:
                scaler_x, scaler_y = SelectedScaler(), SelectedScaler()
                X_train = scaler_x.fit_transform(X_train)
                y_train = scaler_y.fit_transform(y_train)

            # calcualte sample weight based on sample_weight_type
            # type 1: sample weight based on horizontal distance between wells
            if sample_weight_type == 1:
                sample_weight = get_sample_weight(
                    las_name=las_name, las_dict=las_dict, las_lat_lon=las_lat_lon
                )

            # type 2: sample weight based on both horizontal distance between wells and
            # vertical distance in depths, VA (vertical_anisotropy) = 0.2 by default, range: [0, 1]
            # the lower the VA, the more weight on vertical distance, it's a hyperparameter that
            # could be tuned to improve model performance
            elif sample_weight_type == 2:
                sample_weight = get_sample_weight2(
                    las_name=las_name,
                    las_lat_lon=las_lat_lon,
                    las_dict=las_dict,
                    vertical_anisotropy=0.01,
                )

            # 0 or any other value will lead to no sample weight used
            else:
                sample_weight = None

            # fit the model
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except:
                model.fit(X_train, y_train)
                print(
                    "Model does not accept sample weight so sample weight was not used in regular training!"
                )

            # scale test data and predict, and scale back prediction
            X_test = Xy_test.values[:, :-1]
            y_test = Xy_test.values[:, -1:]

            if scaling:
                X_test = scaler_x.transform(X_test)
                y_predict = scaler_y.inverse_transform(
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
            plot_save_path=f"predictions/{TEST_folder}/{model_name}",
            plot_save_format=["png"],  # availabe format: ["png", "html"]
        )

        # plot predicted DTSM vs actual, df_ypred as pd.DataFrame is required for proper plotting
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
            plot_save_path=f"predictions/{TEST_folder}/{model_name}",
            plot_save_format=["png"],  # availabe format: ["png", "html"]
        )

        rmse_all_las_temp.append(rmse_las)
        rmse_all_las.append([las_name, rmse_las])
        print(
            f"{model_name} model with mean rmse so far: {np.mean(rmse_all_las_temp):.2f}"
        )

        # if len(rmse_all_las) >= 2:
        #     break

    rmse_all_las = pd.DataFrame(rmse_all_las, columns=["las_name", model_name])
    rmse_all_las.to_csv(f"predictions/{TEST_folder}/rmse_all_las.csv")

    if scaling:
        return {"model": model, "scaler_x": scaler_x, "scaler_y": scaler_y}
    else:
        return {"model": model}


#%%  TEST 2: split train/test among las files (recommended)

# # choose 7 features/predictors (not including 'DTSM')
# TEST_folder = '6features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']

# # folder to store plots, will create one if not exists
# TEST_folder = '2features_LOOCV_las'
# target_mnemonics = ['DTCO', 'RHOB']

# # folder to store plots, will create one if not exists
# TEST_folder = '5features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'RT']

# # folder to store plots, will create one if not exists
# TEST_folder = '3features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB']

# choose 7 features/predictors (not including 'DTSM')
TEST_folder = "7features_LOOCV_las"
target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

from models.models import model_xgb_7, model_stack

# models = {"XGB_7": model_xgb_7, "MLP_7": model_mlp_7}
# models = {"Stack": model_stack}
models = {"XGB_7": model_xgb_7}

# from models.models import models
time0 = time.time()

model_dict = LOOCV(
    target_mnemonics=target_mnemonics,
    models=models,
    scaling=True,
    TEST_folder=TEST_folder,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon=las_lat_lon,
    sample_weight_type=None,
)

model_dict["info"] = "RobustScaler, no sample weight, rmse_las=9.02"

# pickle model and save
to_pkl(model_dict, f"models/model_{list(models.keys())[0]}.pickle")

print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")

print(f"Prediction results are saved at: predictions/{TEST_folder}")
