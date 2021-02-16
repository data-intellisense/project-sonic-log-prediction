#%% this module is to used test different models
import pickle
import time
import os

import numpy as np
import pandas as pd

import plotly.express as px
import plotly.io as pio
import xgboost
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor as XGB
from tqdm import tqdm

from load_pickle import alias_dict, las_data_DTSM_QC
from plot import plot_crossplot

pio.renderers.default = "browser"

# load customized functions and requried dataset
from util import MeanRegressor, process_las, read_pkl, to_pkl

#%% Batch tuning XGB using RandomizedSearchCV
path = f"predictions/tuning/6_2_despike"

if not os.path.exists(path):
    os.mkdir(path)

feature_model = {
    # "7": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    # "6_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    "6_2": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    # "5_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "PEFZ", "DTSM"],
    # "5_2": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "DTSM"],
    # "5_3": ["DEPTH", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    # "5_4": ["DTCO", "RHOB", "NPHI", "GR", "PEFZ", "DTSM"],
    # "4_1": ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "DTSM"],
    # "4_2": ["DEPTH", "DTCO", "RHOB", "NPHI", "RT", "DTSM"],
    # "4_3": ["DEPTH", "DTCO", "RHOB", "NPHI", "PEFZ", "DTSM"],
    # "3_1": ["DEPTH", "DTCO", "NPHI", "GR", "DTSM"],
    # "3_2": ["DEPTH", "DTCO", "GR", "RT", "DTSM"],
    # "3_3": ["DEPTH", "DTCO", "RHOB", "NPHI", "DTSM"],
    # "3_4": ["DEPTH", "DTCO", "NPHI", "GR", "DTSM"],
    # "3_5": ["DTCO", "NPHI", "GR", "DTSM"],
    # "2_1": ["DEPTH", "DTCO", "NPHI", "DTSM"],
    # "2_2": ["DEPTH", "RHOB", "NPHI", "DTSM"],
    # "1_1": ["DEPTH", "DTCO", "DTSM"],
    # "1_2": ["DTCO", "DTSM"],
}

# create a dictionary to save all the models
model_dict = dict()
time0 = time.time()
scaling = True

for model_name, target_mnemonics in feature_model.items():

    try:
        las_dict = read_pkl(f"{path}/las_dict_{model_name}.pickle")
    except:
        # train_test_split among depth rows
        las_dict = process_las().get_compiled_df_from_las_dict(
            las_data_dict=las_data_DTSM_QC,
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=True,
            outliers_contamination=0.01,
            alias_dict=alias_dict,
            return_dict=True,
            drop_na=True,
        )

        # save the las_dict
        to_pkl(las_dict, f"{path}/las_dict_{model_name}.pickle")

    Xy_ = []
    groups = []
    for las_name, df in las_dict.items():
        Xy_.append(df)
        groups = groups + (list(np.ones(len(df)) * int(las_name[:3])))
    Xy = pd.concat(Xy_)

    if scaling:
        scaler_x, scaler_y = RobustScaler(), RobustScaler()
        X_train = scaler_x.fit_transform(Xy.iloc[:, :-1])
        y_train = scaler_y.fit_transform(Xy.iloc[:, -1:])
    else:
        scaler_x, scaler_y = None, None
        X_train = Xy.iloc[:, :-1].values
        y_train = Xy.iloc[:, -1:].values

    # Baseline model, y_mean and linear regression
    models_baseline = {
        "Mean": MeanRegressor(),
        "MLR": LinearRegression(),
        "RCV": RidgeCV(),
    }

    for model_name_, model in models_baseline.items():
        scores = cross_val_score(
            model,
            X_train,
            y_train,
            cv=5,
            groups=groups,
            scoring="neg_mean_squared_error",
        )
        print(f"{model_name_} rmse:\t{-np.mean(scores):.2f}")
        model_dict[f"rmse_{model_name_}"] = -np.mean(scores)
        if model_name_ == "MLR":
            model_dict[f"estimator_{model_name_}"] = model

    # RandomizedSearchCV to find best hyperparameter combination
    param_distributions = {
        "n_estimators": range(100, 300, 20),
        "max_depth": range(1, 6),
        "min_child_weight": np.arange(0.01, 0.4, 0.01),
        "learning_rate": np.logspace(-3, -1),
        "subsample": np.arange(0.8, 1.02, 0.02),
        "colsample_bytree": np.arange(0.8, 1.02, 0.02),
        "gamma": range(0, 10),
    }

    RandCV = RandomizedSearchCV(
        estimator=XGB(
            tree_method="gpu_hist", objective="reg:squarederror", random_state=42
        ),
        param_distributions=param_distributions,
        n_iter=150,
        scoring="neg_root_mean_squared_error",
        cv=5,
        # cv=GroupKFold(n_splits=5),
        refit=True,
        random_state=42,
        verbose=2,
    )

    RandCV.fit(X=X_train, y=y_train, groups=groups)

    # save all the results
    model_dict["model_name"] = model_name
    model_dict["best_estimator"] = RandCV.best_estimator_
    model_dict["best_params"] = RandCV.best_params_
    model_dict["target_mnemonics"] = list(Xy.columns.values)  # with new mnemonics
    model_dict["scaler_x"] = scaler_x
    model_dict["scaler_y"] = scaler_y
    model_dict["rmse_CV"] = -RandCV.best_score_

    # save all models to pickle file during each iteration, for later prediction
    to_pkl(model_dict, f"{path}/model_xgb_{model_name}.pickle")

    print(
        f"\nCompleted training and saved model in {path} in {time.time()-time0:.1f} seconds!"
    )

    # first, get the best_estimator
    best_estimator = model_dict["best_estimator"]

    # get the feature_importance data
    xgb_feature_importance = [round(i, 3) for i in best_estimator.feature_importances_]
    print(f"Feature importance:\n{Xy.columns} \n{xgb_feature_importance}")

    xgb_feature_importance_df = pd.DataFrame(
        np.c_[
            Xy.columns[:-1].values.reshape(-1, 1),
            np.array(xgb_feature_importance).reshape(-1, 1),
        ],
        columns=["feature", "importance"],
    )

    # plot feature_importance bar
    fig = px.bar(
        xgb_feature_importance_df, x="feature", y="importance", width=1600, height=900
    )
    fig.write_image(f"{path}/xgb_{model_name}_feature_importance.png")

    # calculate y_pred, plot crossplot pred vs true
    # X_train already scaled! Not need to scale again!
    y_predict = best_estimator.predict(X_train).reshape(-1, 1)
    if scaling:
        y_true = scaler_y.inverse_transform(y_train).reshape(-1, 1)
        y_pred = scaler_y.inverse_transform(y_predict).reshape(-1, 1)
    else:
        y_true = np.array(y_train)
        y_pred = np.array(y_predict)

    plot_crossplot(
        y_actual=y_true,
        y_predict=y_pred,
        text=None,
        axis_range=300,
        include_diagnal_line=True,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f"XGB_{model_name} tuning cross plot",
        plot_save_path=path,
        plot_save_format=["png"],
    )
