
from xgboost import XGBRegressor as XGB
from sklearn.linear_model import RidgeCV as RCV 
from sklearn.linear_model import LinearRegression as MLR 
from sklearn.svm import LinearSVR as LSVR 
from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.ensemble import StackingRegressor

estimators = [
    ('MLR', MLR()),
    ('RCV', RCV()),
    #('LSVR', LSVR()),
    # ('RFR', RFR()),
    # ('KNN', KNN()),
    # ('MLP', MLP())
]

model = StackingRegressor(estimators=estimators, final_estimator=LSVR())
models = {'stack_models': model}
# # choose the best model with tuned hyper parameters

# model_7 = XGB(
#     tree_method="hist",
#     objective="reg:squarederror",
#     subsample=0.76,
#     n_estimators=250,
#     min_child_weight=0.02,
#     max_depth=3,
#     learning_rate=0.052,
#     reg_lambda=33,
# )



# # target_mnemonics_3 = ["DTCO", "NPHI", "GR"]
# params = {
#     "subsample": 0.9999999999999999,
#     "n_estimators": 200,
#     "min_child_weight": 0.23,
#     "max_depth": 5,
#     "learning_rate": 0.029470517025518096,
#     "lambda": 68,
# }
# model_3_1 = XGB(**params)

# # target_mnemonics_3 = ["DTCO", "GR", "RT"]
# params = {
#     "subsample": 0.7999999999999999,
#     "n_estimators": 250,
#     "min_child_weight": 0.22,
#     "max_depth": 5,
#     "learning_rate": 0.03906939937054615,
#     "lambda": 18,
# }
# model_3_2 = XGB(**params)

# # target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "RT"]
# params = {
#     "subsample": 0.7999999999999999,
#     "n_estimators": 100,
#     "min_child_weight": 0.39,
#     "max_depth": 5,
#     "learning_rate": 0.0625055192527397,
#     "lambda": 59,
# }
# model_6_1 = XGB(**params)

# #% top part with 6 features
# target_mnemonics_6 = ["DTCO", "NPHI", "RHOB", "GR", "CALI", "PEFZ"]

# params = {
#     "subsample": 0.7,
#     "n_estimators": 200,
#     "min_child_weight": 0.03,
#     "max_depth": 4,
#     "learning_rate": 0.03556480306223128,
#     "lambda": 36,
# }
# model_6_2 = XGB(**params)