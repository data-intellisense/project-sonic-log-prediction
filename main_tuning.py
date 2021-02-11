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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

from pprint import pprint
from sklearn.model_selection import ShuffleSplit
from xgboost import XGBRegressor as XGB
import xgboost

from sklearn.model_selection import RandomizedSearchCV

# from bayes_opt import BayesianOptimization

from plot import plot_crossplot, plot_logs_columns

from load_pickle import alias_dict, las_data_DTSM_QC

# load customized functions and requried dataset
from util import (
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,
    process_las,
    MeanRegressor,
)

pio.renderers.default = "browser"

# change working directory to current file directory if necessary
# path = pathlib.Path(__file__).parent
# os.chdir(path)

#%%  TEST 2: split train/test among las files (recommended)

# # choose 7 features/predictors (not including 'DTSM')
TEST_folder_7 = "7features_LOOCV_las"
target_mnemonics_7 = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

# choose 3 features/predictors (not including 'DTSM')
TEST_folder_3_1 = "3features_LOOCV_las"
target_mnemonics_3_1 = ["DTCO", "NPHI", "RT"]

TEST_folder_2 = "3features_LOOCV_las"
target_mnemonics_2 = ["DTCO", "NPHI"]

TEST_folder = TEST_folder_7
target_mnemonics = target_mnemonics_7

if not os.path.exists(f"predictions/{TEST_folder}"):
    os.mkdir(f"predictions/{TEST_folder}")

target_mnemonics = target_mnemonics + ["DTSM"]  # 'DTSM' is a response variable

las_dict = process_las().get_compiled_df_from_las_dict(
    las_data_dict=las_data_DTSM_QC,
    target_mnemonics=target_mnemonics,
    alias_dict=alias_dict,
    strict_input_output=True,
    add_DEPTH_col=False,
    return_dict=True,
)

Xy = pd.concat([las_dict[k] for k in las_dict.keys()], axis=0)
X_train = Xy.iloc[:, :-1]
y_train = Xy.iloc[:, -1:]


# las_dict_diff = []
# for k in las_dict.keys():
#     temp = las_dict[k].diff(periods=1, axis=0)
#     temp.iloc[0, :] = temp.iloc[1, :]
#     las_dict_diff.append(temp)
# Xy = pd.concat(las_dict_diff, axis=0, ignore_index=True)
# X_train = Xy.values[:, :-1]
# y_train = Xy.values[:, -1:]


scaler_x, scaler_y = RobustScaler(), RobustScaler()
X_train = scaler_x.fit_transform(X_train)
y_train = scaler_y.fit_transform(y_train)


#%% Baseline model, y_mean and linear regression

models = {"Mean-Model": MeanRegressor(), "MLR": LinearRegression(), "RCV": RidgeCV()}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    scores = cross_val_score(
        model, X_train, y_train, cv=10, scoring="neg_mean_squared_error"
    )
    print(f"{model_name} rmse:\t{-np.mean(scores):.2f}")


#%% XGB RandomizedSearchCV tuning

param_distributions = {
    "n_estimators": range(100, 300, 50),
    "max_depth": range(1, 9),
    "min_child_weight": np.arange(0.01, 0.4, 0.01),
    "learning_rate": np.logspace(-3, -1),
    "subsample": np.arange(0.7, 1.0, 0.1),
    "lambda": range(1, 100),
}

time0 = time.time()
RandCV = RandomizedSearchCV(
    estimator=XGB(tree_method="hist", objective="reg:squarederror"),
    param_distributions=param_distributions,
    n_iter=30,
    scoring="neg_mean_squared_error",
    cv=5,
    verbose=2,
)

RandCV.fit(X=X_train, y=y_train)

RandCV.cv_results_
print("best parameters:", RandCV.best_params_)
print(f"Completed training with all models in {time.time()-time0:.1f} seconds!")

#%%
# target_mnemonics = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
params_7 = {
    "subsample": 0.7999999999999999,
    "n_estimators": 250,
    "min_child_weight": 0.31,
    "max_depth": 3,
    "learning_rate": 0.022229964825261943,
    "lambda": 96,
}
model_xgb = XGB(**params_7, tree_method="hist", objective="reg:squarederror")

model_xgb.fit(X=X_train, y=y_train)

xgb_feature_importance = [round(i, 3) for i in model_xgb.feature_importances_]
print(f"Feature importance:\n{df.columns} \n{xgb_feature_importance}")

xgb_feature_importance_df = pd.DataFrame(
    np.c_[
        df.columns[:-1].values.reshape(-1, 1),
        np.array(xgb_feature_importance).reshape(-1, 1),
    ],
    columns=["feature", "importance"],
)
fig = px.bar(xgb_feature_importance_df, x="feature", y="importance")
fig.write_image("readme_resources/xgb_7feature_importance.png")


#%% XGB Bayesian optimization

time0 = time.time()


def xgb_func(**params):
    # params["num_boost_round"]=int(params["num_boost_round"])
    params["max_depth"] = int(params["max_depth"])
    cv_results = xgboost.cv(
        params,
        dtrain=dtrain,
        nfold=5,
        num_boost_round=30,
        early_stopping_rounds=10,
        metrics="rmse",
    )
    return -cv_results.iloc[-1, -2]


param_distributions = {
    "max_depth": (3, 12),
    "min_child_weight": (0.01, 0.5),
    "learning_rate": (0.0001, 0.1),
    "subsample": (0.7, 1.0),
    "lambda": (1, 100),
}


# initialize the BayesOpt
xgb_bayes = BayesianOptimization(
    f=xgb_func, pbounds=param_distributions, verbose=3, random_state=42
)

xgb_bayes.maximize(
    n_iter=30,
    init_points=3,
)
print(xgb_bayes.max)
print(f"finished in {time.time()-time0: .2f} s")

# {'target': -10.141465, 'params': {'lambda': 3.0378649352844422, 'learning_rate': 0.09699399423098323, 'max_depth': 10.491983767203795, 'min_child_weight': 0.11404616423235531, 'subsample': 0.7545474901621302}}
# finished in  1387.96 s

#%% XGB hyperopt

time0 = time.time()


def xgb_func(params):

    cv_results = xgboost.cv(
        params,
        dtrain=dtrain,
        nfold=5,
        num_boost_round=30,
        early_stopping_rounds=10,
        metrics="rmse",
    )
    return {"loss": (cv_results.iloc[-1, -2]), "status": STATUS_OK}


from hyperopt import tpe, hp, fmin, STATUS_OK, Trials

space = {
    # "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.choice("max_depth", range(1, 15)),
    "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.5),
    "learning_rate": hp.loguniform("learning_rate", -4, -1),
    "subsample": hp.uniform("subsample", 0.7, 1.0),
    # "lambda": hp.loguniform("lambda", 0, 2),
}

trials = Trials()
best = fmin(fn=xgb_func, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

print(f"Best:{best}")
print(f"finished in {time.time()-time0: .2f} s")
# best rmse loss: 6.0976406
# Best:{'learning_rate': 0.17921801478908123, 'max_depth': 13, 'min_child_weight': 0.4571378438073825, 'subsample': 0.9917229046738573}
#%% MLP tuning

X_train = RobustScaler().fit_transform(X_train)
y_train = RobustScaler().fit_transform(y_train)

param_distributions = {
    "hidden_layer_sizes": [(100,), (30, 30), (200,), (20, 20, 20)],
    "alpha": [0.00001, 0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.01, 0.1],
}

model = MLP(
    random_state=42,
    learning_rate="adaptive",
    activation="relu",
    max_iter=200,
    early_stopping=True,
)

RandCV = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=10,
    scoring="neg_mean_squared_error",
    verbose=2,
)

RandCV.fit(X=X_train, y=y_train)

print("best parameters:", RandCV.best_params_)
print("Completed training with all models!")

# {'learning_rate_init': 0.01, 'hidden_layer_sizes': (100,), 'alpha': 0.001}

#%% MLP hyperopt

time0 = time.time()


def mlp_func(params):
    MLP = MLPRegressor(
        learning_rate=0.01,
        random_state=42,
        activation="relu",
        max_iter=200,
        early_stopping=True,
    )
    mlp = make_pipeline(RobustScaler(), MLP)
    cv_results = np.mean(cross_val_score(mlp, X_train, y_train, cv=3))
    return {"loss": cv_results, "status": STATUS_OK}


from hyperopt import tpe, hp, fmin, STATUS_OK, Trials

space = {
    # "n_estimators": hp.choice("n_estimators", [100, 200, 300, 400,500,600]),
    "max_depth": hp.choice("max_depth", range(1, 15)),
    "min_child_weight": hp.uniform("min_child_weight", 0.01, 0.5),
    "learning_rate": hp.loguniform("learning_rate", -4, -1),
    "subsample": hp.uniform("subsample", 0.7, 1.0),
    # "lambda": hp.loguniform("lambda", 0, 2),
}

space = {
    "hidden_layer_sizes": hp.choice[(100,), (30, 30), (200,), (20, 20, 20)],
    "alpha": [0.00001, 0.0001, 0.001, 0.01],
    "learning_rate_init": [0.001, 0.01, 0.1],
}

trials = Trials()
best = fmin(fn=xgb_func, space=space, algo=tpe.suggest, max_evals=10, trials=trials)

print(f"Best:{best}")
print(f"finished in {time.time()-time0: .2f} s")

# plot partial dependence
# https://scikit-learn.org/stable/auto_examples/inspection/plot_partial_dependence.html#sphx-glr-auto-examples-inspection-plot-partial-dependence-py

#%% random forest tuning

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
# Number of features to consider at every split
max_features = ["auto", "sqrt"]
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 7, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {
    "n_estimators": n_estimators,
    "max_features": max_features,
    "max_depth": max_depth,
    "min_samples_split": min_samples_split,
    "min_samples_leaf": min_samples_leaf,
    "bootstrap": bootstrap,
}

pprint(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestRegressor()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_best = RandomizedSearchCV(
    estimator=rf,
    param_distributions=random_grid,
    n_iter=20,
    cv=3,
    verbose=2,
    random_state=42,
    n_jobs=1,
)

# Fit the random search model
rf_best.fit(X_train, y_train)
print(rf_best.best_params_)
