#%% import lib
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
from sklearn.linear_model import HuberRegressor as Huber
from sklearn.neural_network import MLPRegressor

from plot import plot_crossplot, plot_logs_columns, plot_outliers

SelectedScaler = RobustScaler

from load_pickle import (
    las_data_DTSM_QC,
    las_lat_lon,
    alias_dict,
    las_depth,
    las_lat_lon,
    test_list,
    TEST_neighbors_20,
    feature_model,
    KMeans_model_3clusters,
    KMeans_model_5clusters,
    KMeans_model_RHOB_2clusters,
    las_data_DTSM_felix,
)

# load customized functions and requried dataset
from util import (
    to_pkl,
    read_pkl,
    get_sample_weight3,
    process_las,
    get_nearest_neighbors,
    predict_zones,
    fit_X_zone,
    pred_y_zone,
)

pio.renderers.default = "browser"


#%%  fit models such as MLP

model_name = "6_2"
algorithm_name = "xgb"

path = f"predictions/evaluate2/felix_data/{model_name}_{algorithm_name}"

if not os.path.exists(path):
    os.mkdir(path)

# import model best_params

model_path = f"predictions/tuning/felix_data_n_iter=200"

for f in glob.glob(f"{model_path}/*.pickle"):
    model = read_pkl(f)
    if model["zone_9"]["model_name"] == model_name:
        print(
            model["zone_9"]["model_name"],
            model["zone_9"]["target_mnemonics"],
            f"{model['zone_9']['rmse_CV']:.3f}",
            "\n",
        )

        params_xgb = model["zone_9"]["best_params"]

model = XGB(
    **params_xgb,
    objective="reg:squarederror",
    tree_method="gpu_hist",
    deterministic_histogram=False,
)


#%% Fit the model

model_dict_zones = dict()

time0 = time.time()

KMeans_model = KMeans_model_3clusters
target_mnemonics = feature_model[model_name]

# TEST_neighbors_20 = ["109-84967b1f42e0_TGS"]
# get the training data
# las_dict = read_pkl(f"data/feature_selected_data/las_dict_{model_name}.pickle")
las_dict = las_data_DTSM_felix
Xy_train = pd.concat(
    [
        val[target_mnemonics]
        for key, val in las_dict.items()
        if key not in TEST_neighbors_20
    ]
)

# zones = []
# for key, df in las_dict.items():
#     if key not in TEST_neighbors_20:
#         z = predict_zones(df=df, cluster_model=KMeans_model).reshape(-1, 1)
#         print(key, np.unique(z))
#         zones.append(z)

# zones = np.concatenate(zones, axis=0)

Xy_test = pd.concat([val for key, val in las_dict.items() if key in TEST_neighbors_20])
print(
    "train and test data length:", len(las_dict) - 17, len(Xy_train), 17, len(Xy_test)
)

# # params_xgb_6_2 = read_pkl(f"models/KFoldCV/model_xgb_{model_name}.pickle")

# params_xgb = {
#     "subsample": 0.94,
#     "n_estimators": 140,
#     "min_child_weight": 0.09,
#     "max_depth": 3,
#     "learning_rate": 0.03906939937054615,
#     "gamma": 5,
#     "colsample_bytree": 0.94,
# }

model_dict, scaler_x, scaler_y = fit_X_zone(
    model=model, cluster_model=KMeans_model, Xy=Xy_train
)
print(model_dict)
# save all related info to a dict
model_dict_zones["model_name"] = model_name
model_dict_zones["zone_model"] = KMeans_model
model_dict_zones["target_mnemonics"] = target_mnemonics
model_dict_zones["scaler_x"] = scaler_x
model_dict_zones["scaler_y"] = scaler_y
model_dict_zones["model_zones"] = model_dict
to_pkl(model_dict_zones, f"{path}/model_{algorithm_name}_{model_name}.pickle")

# model_dict_zones = read_pkl(f"{path}/model_{algorithm_name}_{model_name}.pickle")
# model_dict_zones = read_pkl(f"{path}/model_mlp_6_2.pickle")

print(f"Completed fitting the model in {time.time()-time0} seconds")

#%% LOOCV evaluate
# from models.models import models
time0 = time.time()


def LOOCV_evaluate(
    target_mnemonics=None,
    models=None,
    test_list=None,
    path=None,
    las_data_DTSM=None,
):

    if not os.path.exists(path):
        os.mkdir(path)

    assert isinstance(
        test_list, list
    ), "test_list should be a list of las names for testing."

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    rmse_all_las = []
    rmse_all_las_temp = []

    model_name = models["model_name"]

    # evaluate the model with the pre-defined 107 las files
    for las_name in test_list:

        time1 = time.time()

        Xy_test = las_data_DTSM[las_name].copy()
        Xy_test = process_las().get_df_by_mnemonics(
            df=Xy_test,
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=True,
            alias_dict=alias_dict,
            outliers_contamination=None,  # should not remove outliers when testing!!!
            drop_na=True,  # drop_na should be False when predicting 20 test files
        )

        Xy_test = Xy_test.interpolate()
        X_test = Xy_test.iloc[:, :-1]
        y_test = Xy_test.values[:, -1:]

        # print(X_test.head(), models["target_mnemonics"])

        y_predict = pred_y_zone(
            models=dict(model_name=models["model_zones"]),
            zone_model=models["zone_model"],
            scalers={"x": models["scaler_x"], "y": models["scaler_y"]},
            X_test=X_test,
        )

        # calculate rmse_las
        rmse_las = mean_squared_error(y_test, y_predict, squared=False)

        print(f"{las_name} rmse: {rmse_las:.4f} \trun in {time.time()-time1:.1f} s")

        # plot crossplot to compare y_predict vs y_actual
        plot_crossplot(
            y_actual=y_test,
            y_predict=y_predict,
            include_diagnal_line=True,
            text=None,
            plot_show=False,
            plot_return=False,
            plot_save_file_name=f"{algorithm_name}_{model_name}-{las_name}-Prediction-Crossplot",
            plot_save_path=path,
            plot_save_format=["png"],
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
            plot_save_file_name=f"{algorithm_name}_{model_name}-{las_name}-Prediction-Depth-zone",
            plot_save_path=path,
            plot_save_format=["png"],
        )

        # saving rmse for each las prediction
        rmse_all_las_temp.append(rmse_las)
        rmse_all_las.append([las_name, len(df_ypred), rmse_las])
        rmse_mean = np.mean(rmse_all_las_temp)
        print(f"{model_name} model_dict with mean rmse so far: {rmse_mean:.4f}")

    rmse_all_las = pd.DataFrame(rmse_all_las, columns=["las_name", "rows", "rmse"])
    try:
        rmse_all_las.to_csv(f"{path}/rmse_all_las_{model_name}.csv")
    except:
        rmse_all_las.to_csv(f"{path}/rmse_all_las_{model_name}_temp.csv")
        print(f"Permission denied: {path}/rmse_all_las_{model_name}.csv")

    rmse_corrected = (
        sum((rmse_all_las["rows"] * rmse_all_las["rmse"] ** 2))
        / sum(rmse_all_las["rows"])
    ) ** 0.5
    print(f"{model_name} model_dict with corrected rmse : {rmse_corrected:.2f}")

    return [rmse_mean, rmse_corrected, rmse_all_las]


rmse_LOOCV = LOOCV_evaluate(
    target_mnemonics=model_dict_zones["target_mnemonics"],
    models=model_dict_zones,
    test_list=TEST_neighbors_20,
    path=path,
    las_data_DTSM=las_data_DTSM_QC,
)

# update the model_dict with rmse_LOOCV and save it!
model_dict_zones["rmse_LOOCV_mean"] = rmse_LOOCV[0]
model_dict_zones["rmse_LOOCV_corr"] = rmse_LOOCV[1]
model_dict_zones["rmse_LOOCV_all"] = rmse_LOOCV[2]
to_pkl(model_dict_zones, f"{path}/model_{algorithm_name}_{model_name}.pickle")

print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")
print(f"Prediction results are saved at: {path}")


#%% fit with zone_model and models
