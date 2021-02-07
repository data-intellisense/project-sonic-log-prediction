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

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor as MLP
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor as XGB

from plot import plot_crossplot, plot_logs_columns

from load_pickle import las_data_DTSM_QC, las_lat_lon, alias_dict

# load customized functions and requried dataset
from util import (
    get_alias,
    get_mnemonic,
    get_sample_weight,
    get_sample_weight2,    
    process_las,
    las_name_test
)

pio.renderers.default = "browser"

# change working directory to current file directory
# path = r"C:\Users\Julian Liu\Documents\Project\SPEGCS ML Challenge\project-gcs-datathon2021"
path = pathlib.Path(__file__).parent
os.chdir(path)


#%%  TEST 2: split train/test among las files (recommended)


def train_predict(
    target_mnemonics=None,
    models=None,
    TEST_folder=None,
    las_data_DTSM=None,
    las_lat_lon=None,
    sample_weight_type=2,
):

    if not os.path.exists(f"{path}/predictions/{TEST_folder}"):
        os.mkdir(f"{path}/predictions/{TEST_folder}")

    target_mnemonics = target_mnemonics + ["DTSM"]
    las_dict = dict()

    # get the data that corresponds to terget mnemonics
    for key in las_data_DTSM.keys():
        print(f"Loading {key}")
        df = las_data_DTSM[key]
        
        df = process_las().despike(df, window_size=5)

        df = process_las().get_df_by_mnemonics(
            df=df, target_mnemonics=target_mnemonics, strict_input_output=True, alias_dict=alias_dict
        )

        if (df is not None) and len(df > 1):
            
            # add 'DEPTH' as a feature and rearrange columns
            df['DEPTH']=df.index 
            cols = df.columns.to_list()
            df = df[cols[-1:]+cols[:-1]]

            # df[['LAT','LON']] = las_lat_lon[key]
            # cols = df.columns.to_list()
            # df = df[cols[-3:]+cols[:-3]]

            las_dict[key] = df

    print(
        f"Total {len(las_dict.keys())} las files loaded and total {sum([len(i) for i in las_dict.values()])} rows of data!"
    )

    # evaluate models with Leave One Out Cross Validation (LOOCV)
    # setup recording rmse for each model
    rmse_test = []

    for model_name, model in models.items():

        print(model)
        # reset rmse for each model
        rmse = []
        rmse_ = []

        # create test/train data
        for las_name in las_dict.keys():

            time0 = time.time()

            # use one las file as test data
            Xy_test = las_dict[las_name]

            # create training data dataframe
            Xy_train = pd.concat(
                [las_dict[k] for k in las_dict.keys() if k not in [las_name]], axis=0
            )
            # print('Data Xy_train and Xy_test shape:', Xy_train.shape, Xy_test.shape)

            X_train = Xy_train.values[:, :-1]
            y_train = Xy_train.values[:, -1:]
            
            X_test = Xy_test.values[:, :-1]            
            y_test = Xy_test.values[:, -1:]

            # scale train data
            scaler_x, scaler_y = RobustScaler(), RobustScaler()
            X_train = scaler_x.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)

            # calcualte sample weight based on sample_weight_type
            # type 1: sample weight based on horizontal distance between wells
            if sample_weight_type == 1:
                sample_weight = get_sample_weight(
                    las_name=las_name, las_dict=las_dict, las_lat_lon=las_lat_lon
                )

            # type 2: sample weight based on both horizontal distance between wells and
            # vertical distance in depths, VA (vertical_anisotropy) = 0.2 by default, range: [0, 1]
            # the lower the VA, the more weight on vertical distance, it's a hyperparameter that
            # could be tuned to improve model performance
            elif sample_weight_type == 2:
                sample_weight = get_sample_weight2(
                    las_name=las_name,
                    las_lat_lon=las_lat_lon,
                    las_dict=las_dict,
                    vertical_anisotropy=0.01,
                )

            # 0 or any other value will lead to no sample weight used
            else:
                sample_weight = None

            
            # fit the model
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except:
                model.fit(X_train, y_train)
                print(
                    "Model does not accept sample weight so sample weight was not used in training!"
                )

            
            # scale test data and predict, and scale back prediction
            X_test = scaler_x.transform(X_test)
            y_predict = scaler_y.inverse_transform(model.predict(X_test).reshape(-1, 1))

            # calculate rmse
            rmse_i = mean_squared_error(y_test, y_predict) ** 0.5
            rmse.append([las_name, rmse_i])

            # plot crossplot to compare y_predict vs y_actual
            plot_crossplot(
                y_actual=y_test,
                y_predict=y_predict,
                text=None,
                plot_show=False,
                plot_return=False,
                plot_save_file_name=f"{model_name}-{las_name}-Prediction-Crossplot",
                plot_save_path=f"{path}/predictions/{TEST_folder}/{model_name}",
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
                plot_save_path=f"{path}/predictions/{TEST_folder}/{model_name}",
                plot_save_format=["png"],  # availabe format: ["png", "html"]
            )

            print(
                f"Completed fitting with {model_name} model in {time.time()-time0:.2f} seconds"
            )
            print(f"{las_name}, rmse: {rmse[-1][-1]:.2f}")

            rmse_.append(rmse_i)
            print(f'Mean RMSE so far: {np.mean(rmse_):.2f}')
            # print("Only trained one stage! Remove 'break' to train all stages!")
            # break

        rmse = pd.DataFrame(rmse, columns=["las_name", model_name])
        rmse_test.append(rmse)
        

    # # covnert rmse_test to pd.DataFrame and save to .csv
    rmse_test = pd.concat(rmse_test, axis=1)
    rmse_test.to_csv(f"{path}/predictions/{TEST_folder}/rmse_test.csv")

    return rmse_test, [model, scaler_x, scaler_y]


#%%  TEST 2: split train/test among las files (recommended)

# # choose 7 features/predictors (not including 'DTSM')
# TEST_folder = '6features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']

# # folder to store plots, will create one if not exists
# TEST_folder = '2features_LOOCV_las'
# target_mnemonics = ['DTCO', 'RHOB']

# # folder to store plots, will create one if not exists
# TEST_folder = '5features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'RT']

# # folder to store plots, will create one if not exists
# TEST_folder = '3features_LOOCV_las'
# target_mnemonics = ['DTCO', 'NPHI', 'RHOB']

# choose 8 features/predictors (not including 'DTSM')
TEST_folder = "6features_LOOCV_las"
target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]

from models.models import model_mlp_7 as models

# from models.models import models

rmse_test, *model = train_predict(
    target_mnemonics=target_mnemonics,
    models=models,
    TEST_folder=TEST_folder,
    las_data_DTSM=las_data_DTSM_QC,
    las_lat_lon=las_lat_lon,
    sample_weight_type=2,
)

# # pickle model and save
# with open(f"models/model_{TEST_folder}.pickle", "wb") as f:
#     pickle.dump(model, f)

print("Completed training with all models!")

rmse_test_ = dict()
for col in rmse_test.columns[1:]:
    rmse_test_[col] = rmse_test[col].mean()
print(rmse_test_)

print(f"Prediction results are saved at: {path}/predictions/{TEST_folder}")
