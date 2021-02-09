from xgboost import XGBRegressor as XGB
from sklearn.linear_model import RidgeCV as RCV
from sklearn.linear_model import LinearRegression as MLR
from sklearn.svm import LinearSVR as LSVR
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.ensemble import StackingRegressor

estimators = [
    ("MLR", MLR()),
    ("RCV", RCV()),
    # ('LSVR', LSVR()),
    # ('RFR', RFR()),
    # ('KNN', KNN()),
    # ('MLP', MLP())
]

# model = StackingRegressor(estimators=estimators, final_estimator=LSVR())
# models = {'stack_models': model}
# # choose the best model with tuned hyper parameters

# neural network model
params_mlp_7 = {
    "learning_rate_init": 0.01,
    "hidden_layer_sizes": (100,),
    "alpha": 0.001,
}
model_mlp_7 = MLP(
    random_state=42,
    learning_rate="adaptive",
    activation="relu",
    max_iter=200,
    early_stopping=True,
    **params_mlp_7
)

# using randomizedsearchcv, lowest rmse, 9.85
# target_mnemonics = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT", "PEFZ"]
# params_xgb_7 = {'subsample': 0.7999999999999999, 'n_estimators': 250,
#             'min_child_weight': 0.31, 'max_depth': 3,
#             'learning_rate': 0.022229964825261943, 'lambda': 96}

# xgb model based the difference of 7 features, not scaled
# params_xgb_7 = {
#     "subsample": 0.7999999999999999,
#     "n_estimators": 150,
#     "min_child_weight": 0.08,
#     "max_depth": 2,
#     "learning_rate": 0.06866488450042998,
#     "lambda": 72,
# }

# xgb model based the difference of 7 features, RobustScaler,  rmse=9.025
params_xgb_7 = {
    "subsample": 0.8999999999999999,
    "n_estimators": 150,
    "min_child_weight": 0.08,
    "max_depth": 3,
    "learning_rate": 0.03906939937054615,
    "lambda": 31,
}

model_xgb_7 = XGB(**params_xgb_7, tree_method="hist", objective="reg:squarederror")

# xgb model based the difference of 7 features, not scaled
# params_xgb_d7 = {
#     "subsample": 0.9999999999999999,
#     "n_estimators": 150,
#     "min_child_weight": 0.38,
#     "max_depth": 4,
#     "learning_rate": 0.06866488450042998,
#     "lambda": 97,
# }

# xgb model based the difference of 7 features, RobustScaler
params_xgb_d7 = {
    "subsample": 0.8999999999999999,
    "n_estimators": 100,
    "min_child_weight": 0.09,
    "max_depth": 3,
    "learning_rate": 0.1,
    "lambda": 67,
}
model_xgb_d7 = XGB(**params_xgb_d7, tree_method="hist", objective="reg:squarederror")

# using Bayesian Optimization, lowest cv rmse 10.1, loocv las rmse: 10.69
params_xgb_7_1 = {
    "lambda": 3.0378649352844422,
    "learning_rate": 0.09699399423098323,
    "max_depth": 10,
    "min_child_weight": 0.11404616423235531,
    "subsample": 0.7545474901621302,
}
model_xgb_7_1 = XGB(**params_xgb_7_1, tree_method="hist", objective="reg:squarederror")


# using hyperopt, lowest rmse 6.1, loocv las rmse: 11.56 (with or without sample weight)
params_xgb_7_2 = {
    "learning_rate": 0.17921801478908123,
    "max_depth": 13,
    "min_child_weight": 0.4571378438073825,
    "subsample": 0.9917229046738573,
}
model_xgb_7_2 = XGB(**params_xgb_7_2, tree_method="hist", objective="reg:squarederror")


params_xgb_6_0 = {
    "subsample": 0.7999999999999999,
    "n_estimators": 250,
    "min_child_weight": 0.23,
    "max_depth": 7,
    "learning_rate": 0.03556480306223128,
    "lambda": 81,
}

model_xgb_6_0 = XGB(**params_xgb_6_0, tree_method="hist", objective="reg:squarederror")

# ['DTCO', 'NPHI']
params_xgb_2 = {
    "subsample": 1,
    "n_estimators": 250,
    "min_child_weight": 0.36,
    "max_depth": 7,
    "learning_rate": 0.02,
    "lambda": 56,
}

model_xgb_2 = XGB(**params_xgb_2, tree_method="hist", objective="reg:squarederror")


# target_mnemonics_3 = ["DTCO", "NPHI", "GR"]
params_xgb_3_1 = {
    "subsample": 0.9999999999999999,
    "n_estimators": 200,
    "min_child_weight": 0.23,
    "max_depth": 5,
    "learning_rate": 0.029470517025518096,
    "lambda": 68,
}
model_xgb_3_1 = XGB(**params_xgb_3_1)

# target_mnemonics_3 = ["DTCO", "GR", "RT"]
params_xgb_3_2 = {
    "subsample": 0.7999999999999999,
    "n_estimators": 250,
    "min_child_weight": 0.22,
    "max_depth": 5,
    "learning_rate": 0.03906939937054615,
    "lambda": 18,
}
model_xgb_3_2 = XGB(**params_xgb_3_2)

# target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT"]
params_xgb_6_1 = {
    "subsample": 0.7999999999999999,
    "n_estimators": 100,
    "min_child_weight": 0.39,
    "max_depth": 5,
    "learning_rate": 0.0625055192527397,
    "lambda": 59,
}
model_xgb_6_1 = XGB(**params_xgb_6_1)

#% top part with 6 features
target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "PEFZ"]

params_xgb_6_2 = {
    "subsample": 0.7,
    "n_estimators": 200,
    "min_child_weight": 0.03,
    "max_depth": 4,
    "learning_rate": 0.03556480306223128,
    "lambda": 36,
}
model_xgb_6_2 = XGB(**params_xgb_6_2)