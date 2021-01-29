#%% import library
from itertools import product
from copy import deepcopy
from xgboost import XGBRFRegressor as xgb 

params_xgb = {
    "n_estimators": (100, 300),
    "max_depth": (3, 12),
    "min_child_weight": (0.01, 0.3),
    "learning_rate": (0.001, 0.1),
    "subsample": (0.7, 1.0),
}

