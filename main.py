#%% this module is to used test different models
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import time
from sklearn.preprocessing import RobustScaler
import random 

# models
from sklearn.linear_model import RidgeCV as RCV
from sklearn.neighbors import KNeighborsRegressor as KNN
from sklearn.svm import LinearSVR as LSVR
from sklearn.ensemble import GradientBoostingRegressor as GBR 
from sklearn.neural_network import MLPRegressor as MLP
from xgboost import XGBRegressor as XGB
from sklearn.ensemble import StackingRegressor as Stack

from plot import plot_logs_columns, plot_crossplot
import plotly.express as px 
import plotly.io as pio
import pathlib 

from util import alias_dict, get_alias, get_mnemonic, CV_weighted, process_las, las_lat_lon, get_distance

pio.renderers.default='browser'

path = pathlib.Path(__file__).parent
with open(f'{path}/data/las_data_DTSM.pickle', 'rb') as f:
    las_data_DTSM = pickle.load(f)


#%%  TEST 2: split train/test among las files (recommended)

def train_predict(target_mnemonics=None,
                  models=None,
                  TEST_folder=None,
                  las_data_DTSM=None,  
                  las_lat_lon=None,
                  use_sample_weight=True                                        
                  ):

    if not os.path.exists(f'{path}/predictions/{TEST_folder}'):
        os.mkdir(f'{path}/predictions/{TEST_folder}')

    # choose 7 features/predictors (not including 'DTSM')

    target_mnemonics = target_mnemonics + ['DTSM'] # 'DTSM' is a response variable
    las_dict = dict()

    # get the data that corresponds to terget mnemonics
    for key in las_data_DTSM.keys():
        print(f'Loading {key}')
        df_ = process_las().get_df_by_mnemonics(las_data_DTSM[key], target_mnemonics=target_mnemonics, strict_input_output=True)

        if (df_ is not None) and len(df_>1):
            las_dict[key] = df_

    print(f'Total {len(las_dict.keys())} las files loaded and total {sum([len(i) for i in las_dict.values()])} rows of data!')

    # evaluate models with Leave One Out Cross Validation (LOOCV)


    # setup recording rmse for each model
    rmse_test = []

    for model_name, model in models.items():
        
        # reset rmse for each model
        rmse = []
        

        # create test/train data
        for las_name in las_dict.keys():

            time0 = time.time()

            # use one las file as test data
            Xy_test = las_dict[las_name]

            # create training data dataframe 
            Xy_train = pd.concat([las_dict[k] for k in las_dict.keys() if k not in [las_name]], axis=0)
            # print('Data Xy_train and Xy_test shape:', Xy_train.shape, Xy_test.shape)        

            
            X_train = Xy_train.values[:, :-1]
            y_train = Xy_train.values[:, -1:]
            X_test  = Xy_test.values[:, :-1]
            y_test = Xy_test.values[:, -1:]

            # scale train data
            scaler_x, scaler_y = RobustScaler(), RobustScaler()
            X_train = scaler_x.fit_transform(X_train)
            y_train = scaler_y.fit_transform(y_train)

            # fit the model
            if use_sample_weight:
                # get sample weight = 1/distance between target las and remaining las
                sample_weight = []
                for k in las_dict.keys():
                    if k not in [las_name]:
                        sample_weight = sample_weight +[1/get_distance(las_lat_lon[las_name], las_lat_lon[k])]*len(las_dict[k])
                sample_weight = np.array(sample_weight).reshape(-1,1)
                
                reg = model.fit(X_train, y_train, sample_weight=sample_weight)
            else:
                reg = model.fit(X_train, y_train)

            # scale test data and predict, and scale back prediction
            X_test = scaler_x.transform(X_test)
            y_predict = scaler_y.inverse_transform(reg.predict(X_test).reshape(-1,1))   
            
            # calculate rmse
            rmse.append(mean_squared_error(y_test, y_predict)**.5)

            # plot crossplot to compare y_predict vs y_actual
            plot_crossplot(y_actual=y_test, 
                        y_predict=y_predict,
                        text=None,
                        plot_show=False,
                        plot_return=False,
                        plot_save_file_name=f'{model_name}-{las_name}-Prediction-Crossplot',
                        plot_save_path=f'{path}/predictions/{TEST_folder}/{model_name}',
                        plot_save_format=['png']
            )
        
            # plot predicted DTSM vs actual, df_ypred as pd.DataFrame is required for proper plotting
            df_ypred = pd.DataFrame(np.c_[Xy_test.index.values.reshape(-1,1), y_predict.reshape(-1,1)], columns=['Depth', 'DTSM_Pred'])
            plot_logs_columns(
                df=Xy_test,
                DTSM_pred=df_ypred,
                well_name=las_name,
                plot_show=False,
                plot_return=False,
                plot_save_file_name=f'{model_name}-{las_name}-Prediction-Depth',
                plot_save_path=f'{path}/predictions/{TEST_folder}/{model_name}',
                plot_save_format=['png']  # availabe format: ["png", "html"]
            )

            print(f'Finished fitting with {model_name} model in {time.time()-time0:.2f} seconds')

        rmse = pd.DataFrame(rmse, columns=[model_name])
        rmse_test.append(rmse)

    # # covnert rmse_test to pd.DataFrame and save to .csv
    rmse_test = pd.concat(rmse_test, axis=1) 
    rmse_test.to_csv(f'{path}/predictions/{TEST_folder}/rmse_test.csv')

print('Completed training with all models!')

#%%  TEST 2: split train/test among las files (recommended)

# folder to store plots, will create one if not exists
# choose 7 features/predictors (not including 'DTSM')
TEST_folder = '7features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']

# folder to store plots, will create one if not exists
TEST_folder = '2features_LOOCV_las'
target_mnemonics = ['DTCO', 'RHOB']


# folder to store plots, will create one if not exists
TEST_folder = '5features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'RT']


# folder to store plots, will create one if not exists
TEST_folder = '3features_LOOCV_las'
target_mnemonics = ['DTCO', 'NPHI', 'RHOB']

# assemble all models in a dictionary
models = {
        # 'RCV': RCV(), 
        #'LSVR': LSVR(epsilon=0.1),
        #'KNN': KNN(n_neighbors=10),
        # 'HGBR': HGBR(),
        # 'MLP': MLP(hidden_layer_sizes=(20,)),
        'XGB': XGB(tree_method='hist', objective='reg:squarederror', n_estimators=100),
}

train_predict(target_mnemonics=target_mnemonics,
              models=models,
              TEST_folder=TEST_folder,
              las_data_DTSM=las_data_DTSM,                                    
              las_lat_lon=las_lat_lon, 
              use_sample_weight=True
              )