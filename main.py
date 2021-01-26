#%% this module is to test deifferent models
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

from util import get_alias, get_mnemonic, CV_weighted, process_las

pio.renderers.default='browser'

with open('data/las_data_DTSM.pickle', 'rb') as f:
    las_dict = pickle.load(f)

with open('data/alias_dict.pickle', 'rb') as f:
    alias_dict = pickle.load(f)


#%% TEST: get_mnemonics_from_df and create data

if __name__ == '__main__':

    # load las_data_DTSM
    with open('data/las_data_DTSM.pickle', 'rb') as f:
        las_data_DTSM = pickle.load(f)

    # 7 features
    target_mnemonics = ['DTCO', 'NPHI','RHOB', 'GR', 'CALI', 'RT', 'PEFZ', 'DTSM']        
    df = None
    key_list = []
    for key in las_data_DTSM.keys():
        print(f'Loading {key}')
        df_ = process_las().get_df_by_mnemonics(las_data_DTSM[key], target_mnemonics=target_mnemonics, strict_input_output=True)
        if df_ is not None:
            key_list.append(key)
            if df is None:
                df = df_
            else:
                df = pd.concat([df, df_], axis=0)
    print(f'Total {len(key_list)} las loaded and total {len(df)} rows of data!')
    with open('data/las_data_7features.pickle', 'wb') as f:
        pickle.dump(df, f)


#%% fit models with 7 features dataset

X = df.iloc[:, :-1]
y = df.iloc[:, -1:]

print('Data X and y shape:', X.shape, y.shape)

models = {
        'RCV': RCV(), 
        'LSVR': LSVR(epsilon=0.1),
        'KNN': KNN(n_neighbors=10),
        'GBR': GBR(),
        'MLP': MLP(hidden_layer_sizes=(10,)),
        'XGB': XGB(tree_method='hist', objective='reg:squarederror', n_estimators=100),
}

for name, model in models.items():
    time0 = time.time()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
    scaler_x, scaler_y = RobustScaler(), RobustScaler()
    X_train = scaler_x.fit_transform(X_train)
    y_train = scaler_y.fit_transform(y_train)

    reg = model.fit(X_train, y_train)

    X_test = scaler_x.transform(X_test)
    y_predict = scaler_y.inverse_transform(reg.predict(X_test).reshape(-1,1))   
    
    # not the correct way, should scale train and test data seprately
    X_ = scaler_x.fit_transform(X)
    y_ = scaler_x.fit_transform(y)
    text = f'{name} - rmse_cv: {CV_weighted(reg, X_, y_):.2f}'
    
    # plot crossplot
    plot_crossplot(y_actual=y_test.values, 
                y_predict=y_predict,
                text=text,
                plot_show=True,
                plot_return=False,
                plot_save_file_name=f'Prediction-{name}',
                plot_save_path='predictions/7features',
                plot_save_format=['png'])
   
    print(f'Finished fitting with {name} model in {time.time()-time0:.2f} seconds')