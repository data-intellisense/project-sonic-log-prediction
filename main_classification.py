#%% import lib
import numpy as np
import pandas as pd

from util import process_las, to_pkl, read_pkl, assign_rock_type
from xgboost import XGBClassifier as XGBC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings("ignore")
from load_pickle import (
    las_data_DTSM_QC,
    feature_model,
    alias_dict,
    las_data_DTSM_QC2,
    rock_info,
)

model_name = "6_2"

target_mnemonics = feature_model[model_name]
# info_ = pd.read_csv("data/curve_info_to_QC2.csv", index_col=["WellName"])

# create rock_info.pickle
# info = info_[["rock_median", "rock_type", "rock_type2"]][info_["drop_las"] != 1]
# to_pkl(info, "models/rock_info.pickle")


# create las_data_DTSM_QC2.pickle
# las_data_DTSM_QC2 = dict()
# for key, df in las_data_DTSM_QC.items():
#     for key2, row in info.iterrows():
#         if key == key2:
#             if not pd.isnull(row["Top Depth"]):
#                 df = df[df.index >= row["Top Depth"]]
#             if not pd.isnull(row["Btm Depth"]):
#                 df = df[df.index <= row["Btm Depth"]]
#             if not pd.isnull(row["drop_las"]):
#                 df = None
#         print(key2)
#     if df is not None:
#         las_data_DTSM_QC2[key] = df

# # check length
# len(pd.concat([v for v in las_data_DTSM_QC.values()]))  # 225, 1910753
# len(pd.concat([v for v in las_data_DTSM_QC2.values()]))  # 219, 1834979
# len(las_data_DTSM_QC)
# len(las_data_DTSM_QC2)
# to_pkl(las_data_DTSM_QC2, f"data/las_data_DTSM_QC2.pickle")

#%% build data for classification


Xy = []
for key in rock_info.index:

    print(key)

    df_ = process_las().get_df_by_mnemonics(
        df=las_data_DTSM_QC2[key],
        target_mnemonics=target_mnemonics,
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        outliers_contamination=None,  # should not remove outliers when testing!!!
        drop_na=True,  # drop_na should be False when predicting 20 test files
    )

    df_ = assign_rock_type(df=df_, las_name=key, info=rock_info)

    Xy.append(df_)

    if "064" in key:
        print(df_.sample(10))

Xy = pd.concat(Xy)
print(Xy.sample(10))

np.unique(Xy["rock_type"])
len(Xy["rock_type"])  #  747459
sum(Xy["rock_type"])  #  222945, 30% rock type [1] out ot type [0, 1]

X = Xy.iloc[:, :-2]
y = Xy.iloc[:, -1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#%% CV tune hyperparameters
# RandomizedSearchCV to find best hyperparameter combination
param_distributions = {
    "n_estimators": range(100, 300, 20),
    "max_depth": range(1, 9),
    "min_child_weight": np.arange(0.01, 0.4, 0.01),
    "learning_rate": np.logspace(-3, -1),
    "subsample": np.arange(0.8, 1.02, 0.02),
    "colsample_bytree": np.arange(0.8, 1.02, 0.02),
    "gamma": range(0, 10),
}

RandCV = RandomizedSearchCV(
    estimator=XGBC(tree_method="gpu_hist", eval_metric="logloss", random_state=42),
    param_distributions=param_distributions,
    n_iter=60,
    scoring="accuracy",
    cv=5,
    refit=True,
    random_state=42,
    verbose=2,
)

RandCV.fit(X=X_train, y=y_train)

model_dict = dict()
model_dict["best_estimator"] = RandCV.best_estimator_
model_dict["best_params"] = RandCV.best_params_
model_dict["rmse_CV"] = RandCV.best_score_

# to_pkl(model_dict, "models/model_xgb_rock_classification.pickle")

print("Completed CV tuning!")

#% predict

# xgbc_params = dict(n_estimators=100, max_depth=3, learning_rate=0.01)
# xgbc = XGBC(**xgbc_params)

xgbc = RandCV.best_estimator_
xgbc.fit(X_train, y_train)
# evals_result = xgbc.evals_result()
y_pred = xgbc.predict(X_test)

c_report = classification_report(y_test, y_pred)
print(c_report)

model_dict["c_report"] = c_report
to_pkl(model_dict, f"models/model_xgb_rock_class_{model_name}.pickle")