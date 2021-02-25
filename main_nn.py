#%% import lib
# import tensorflow as tf

from numpy.random import seed

seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
import seaborn as sb
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.signal import medfilt
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.callbacks import ModelCheckpoint
import keras
import warnings

from util import KMeans_model_5clusters

warnings.filterwarnings("ignore")

#%% A simple fully connected neural network is used for prediction
model_1 = Sequential(
    [
        Dense(24, activation="relu", input_shape=(27,)),
        Dense(12, activation="relu"),
        Dense(2, activation="sigmoid"),
    ]
)


model_1.compile(optimizer="adam", loss="mse", metrics=["mae"])
np.random.seed(1)


save_model_name = "NN_Model"
model_checkpoint = ModelCheckpoint(
    save_model_name, monitor="loss", mode="min", save_best_only=True, verbose=1
)

# hist_1 = model_1.fit(
#     x_train, y_train, batch_size=10, epochs=9, callbacks=[model_checkpoint]
# )

path = f"predictions/tuning/6_2_nn"
KMeans_model = KMeans_model_5clusters

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

time0 = time.time()

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
            outliers_contamination=None,
            alias_dict=alias_dict,
            return_dict=True,
            drop_na=True,
        )

        # save the las_dict
        to_pkl(las_dict, f"{path}/las_dict_{model_name}.pickle")

    Xy_ = []
    # groups = []
    for las_name, df in las_dict.items():
        Xy_.append(df)
    Xy = pd.concat(Xy_)

    # scale before zoning
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train_ = scaler_x.fit_transform(Xy.iloc[:, :-1])
    y_train_ = scaler_y.fit_transform(Xy.iloc[:, -1:])

    # predict zones
    zones = predict_zones(df=Xy, cluster_model=KMeans_model)
    zone_ids = np.unique(zones)

    model_dict_zones = dict()
    for zone_id in zone_ids:

        # reset model_dict for each zone_id
        model_dict = dict()

        # get data for each specific zone and scale the data

        # scale after zoning
        # scaler_x, scaler_y = RobustScaler(), RobustScaler()
        # X_train = scaler_x.fit_transform(Xy.iloc[:, :-1][zones == zone_id])
        # y_train = scaler_y.fit_transform(Xy.iloc[:, -1:][zones == zone_id])

        X_train = X_train_[zones == zone_id]
        y_train = y_train_[zones == zone_id]

        print(f"zone_id: {zone_id}, data rows: {len(X_train)}", "\n", "*" * 70)

        # Baseline model, y_mean and linear regression
        models_baseline = {"MLR": LinearRegression()}

        for model_name_, model in models_baseline.items():
            scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=5,
                scoring="neg_root_mean_squared_error",
            )
            print(f"{model_name_} rmse:\t{-np.mean(scores):.2f}")
            model_dict[f"rmse_{model_name_}"] = -np.mean(scores)
            # if model_name_ == "MLR":
            #     model_dict[f"estimator_{model_name_}"] = model