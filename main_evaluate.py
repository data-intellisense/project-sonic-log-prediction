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
    TEST_folder=None,
    las_data_DTSM=None,
    las_lat_lon=None,
    sample_weight_type=None,
):

    if not os.path.exists(f"predictions/{TEST_folder}"):
        os.mkdir(f"predictions/{TEST_folder}")

    # target_mnemonics = target_mnemonics + ["DTSM"]

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    # setup recording rmse_las for each model_dict
    rmse_all_las = []
    rmse_all_las_temp = []

    # Xy = process_las().get_compiled_df_from_las_dict(
    #     las_data_dict=las_data_DTSM_QC,
    #     target_mnemonics=target_mnemonics,
    #     new_mnemonics=["DEPTH"],
    #     log_mnemonics=["RT"],
    #     strict_input_output=True,
    #     alias_dict=alias_dict,
    #     drop_na=True,
    #     return_dict=False,
    # )

    # # scale train data
    # if scaling:
    #     scaler_x, scaler_y = SelectedScaler(), SelectedScaler()
    #     _ = scaler_x.fit_transform(Xy.values[:, :-1])
    #     _ = scaler_y.fit_transform(Xy.values[:, -1:])

    print("models:", models)
    # evaluate the model_dict again 107 las files
    for _, las_name in test_list.itertuples():
        Xy_test = las_data_DTSM_QC[las_name].copy()
        Xy_test = process_las().get_df_by_mnemonics(
            df=Xy_test,
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=True,
            alias_dict=alias_dict,
            drop_na=True,
        )

        # reset rmse_las for each model_dict
        rmse_las = []
        y_predict_models = []

        for model_name, model_dict in models.items():

            time0 = time.time()

            # scale test data and predict, and scale back prediction
            X_test = Xy_test.values[:, :-1]
            y_test = Xy_test.values[:, -1:]

            if scaling:
                X_test = scalers[0].transform(X_test)
                y_predict = scalers[1].inverse_transform(
                    model_dict.predict(X_test).reshape(-1, 1)
                )
            else:
                y_predict = model_dict.predict(X_test).reshape(-1, 1)

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
            f"{model_name} model_dict with mean rmse so far: {np.mean(rmse_all_las_temp):.2f}"
        )

    rmse_all_las = pd.DataFrame(rmse_all_las, columns=["las_name", model_name])
    rmse_all_las.to_csv(f"predictions/{TEST_folder}/rmse_all_las_{model_name}.csv")

    return np.mean(rmse_all_las_temp)  # final rmse for all 107 las


#%%  TEST 2: split train/test among las files (recommended)


for model in glob.glob("models/*.pickle"):

    # load the model_dict
    model_dict = read_pkl(model)

    print(
        model_dict["model_name"], model_dict["target_mnemonics"], model_dict["rmse_CV"]
    )
    print("\n")

    model_name = model_dict["model_name"]
    TEST_folder = f"LOOCV_evaluate_{model_name}"
    models = {f"XGB_{model_name}": model_dict["best_estimator"]}

    # from models.models import models
    time0 = time.time()

    rmse_LOOCV = LOOCV_evaluate(
        target_mnemonics=model_dict["target_mnemonics"],
        models=models,
        scaling=True,
        scalers=[model_dict["scaler_x"], model_dict["scaler_y"]],
        TEST_folder=TEST_folder,
        las_data_DTSM=las_data_DTSM_QC,
        las_lat_lon=las_lat_lon,
        sample_weight_type=None,
    )

    # update the model_dict with rmse_LOOCV and save it!
    model_dict["rmse_LOOCV"] = rmse_LOOCV
    to_pkl(model_dict, f"models/model_xgb_{model_name}.pickle")

    print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")
    print(f"Prediction results are saved at: predictions/{TEST_folder}")

#%% check out the models

temp = []
for model in glob.glob("models/*.pickle"):

    # load the model_dict
    model_dict = read_pkl(model)
    temp.append(
        [
            model_dict["model_name"],
            model_dict["rmse_LOOCV"],
            model_dict["rmse_CV"],
            model_dict["target_mnemonics"],
        ]
    )

print("\n")
temp = pd.DataFrame(temp, columns=["model_name", "rmse_LOOCV", "rmse_CV", "mnemonics"])
print(temp.sort_values(by=["rmse_LOOCV"], ascending=True))
