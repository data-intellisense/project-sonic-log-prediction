#%% this module is to used test different models
import pickle
import time
from pprint import pprint

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import xgboost
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor as XGB

from load_pickle import alias_dict, las_data_DTSM_QC
from plot import plot_crossplot

pio.renderers.default = "browser"

# load customized functions and requried dataset
from util import MeanRegressor, process_las, read_pkl, to_pkl

#%% Batch tuning XGB using RandomizedSearchCV

mnemonic_dict = {
    '7':   ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"],
    '6_1': ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "DTSM"],
    '6_2': ["DTCO", "RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    '5_1': ["DTCO", "RHOB", "NPHI", "GR", "PEFZ", "DTSM"],
    '5_2': ["DTCO", "RHOB", "NPHI", "GR", "RT", "DTSM"],    
    '5_3': ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTSM"],
    '4_1': ["DTCO", "RHOB", "NPHI", "GR", "DTSM"],
    '4_2': ["DTCO", "RHOB", "NPHI", "RT", "DTSM"],
    '4_3': ["DTCO", "RHOB", "NPHI", "PEFZ", "DTSM"],
    '3_1': ["DTCO", "NPHI", "GR", "DTSM"],
    '3_2': ["DTCO", "GR", "RT", "DTSM"],
    '3_3': ["DTCO", "RHOB", "NPHI", "DTSM"],
    '2_1': ["DTCO", "NPHI", "DTSM"],
    '2_2': ["RHOB", "NPHI", "DTSM"],
    # "DTCO" as response,for well 6 and 8, to fix DTCO
    # "DTCO_5": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "DTCO"],
    # "DTCO_6": ["RHOB", "NPHI", "GR", "CALI", "PEFZ", "RT", "DTCO"],
}

# create a dictionary to save all the models
model_dict = dict()

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

    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(Xy.iloc[:, :-1])
    y_train = scaler_y.fit_transform(Xy.iloc[:, -1:])

    # Baseline model, y_mean and linear regression
    models_baseline = {"Mean": MeanRegressor(), "MLR": LinearRegression(), "RCV": RidgeCV()}

    for model_name_, model in models_baseline.items():
        scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        print(f"{model_name_} rmse:\t{-np.mean(scores):.2f}")
        model_dict[f'rmse_{model_name_}'] = -np.mean(scores)
        if model_name_=='MLR':
            model_dict[f'estimator_{model_name_}'] = model

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
        estimator=XGB(tree_method="gpu_hist", objective="reg:squarederror", random_state=42),
        param_distributions=param_distributions,
        n_iter=150,
        scoring="neg_root_mean_squared_error",
        cv=5,
        refit=True,
        verbose=2,
    )

    RandCV.fit(X=X_train, y=y_train)   

    # save all the results
    model_dict['model_name'] = model_name
    model_dict['best_estimator'] = RandCV.best_estimator_
    model_dict['target_mnemonics'] = list(Xy.columns.values) # with new mnemonics
    model_dict['scaler_x'] = scaler_x
    model_dict['scaler_y'] = scaler_y    
    model_dict['rmse_CV'] = -RandCV.best_score_
    
    # save all models to pickle file during each iteration, for later prediction
    to_pkl(model_dict, f"predictions/tuning/model_xgb_{model_name}.pickle")

    print(f"\nCompleted training and saved model in {time.time()-time0:.1f} seconds!")

    # first, get the best_estimator
    best_estimator = model_dict['best_estimator']

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
    fig = px.bar(xgb_feature_importance_df, x="feature", y="importance", width=1600, height=900)
    fig.write_image(f"readme_resources/xgb_{model_name}_feature_importance.png")

    # calculate y_pred, plot crossplot pred vs true        
    y_predict = best_estimator.predict(X_train).reshape(-1,1)

    plot_crossplot(
        y_actual=scaler_y.inverse_transform(y_train).reshape(-1,1),
        y_predict=scaler_y.inverse_transform(y_predict).reshape(-1,1),
        text=None,
        axis_range=300,
        include_diagnal_line=True,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=f'XGB_{model_name} tuning cross plot',
        plot_save_path='predictions/tuning',
        plot_save_format= ["png"],
    )
