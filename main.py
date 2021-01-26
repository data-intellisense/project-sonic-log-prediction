#%% this module is to test deifferent models
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from util import get_features_df, get_alias, get_mnemonic, CV_weighted, get_df_from_mnemonics
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
pio.renderers.default='browser'

with open('data/las_data_DTSM.pickle', 'rb') as f:
    las_dict = pickle.load(f)

with open('data/alias_dict.pickle', 'rb') as f:
    alias_dict = pickle.load(f)

#%% create simple DTSM/DTCO model

# # create ['DTCO'] as X and ['DTSM'] as y
# Xlables = ['DTCO']
# ylabels = ['DTSM']
# X = None
# y = None

# for key, df in las_dict.items():
#     xlabels_alias = []
#     ylabels_alias = []

#     for label in Xlables:
#         for col in df.columns:
#             mnemonic = get_mnemonic(col, alias_dict=alias_dict)
#             if (mnemonic==label) and (mnemonic not in xlabels_alias):            
#                xlabels_alias.append(col)

#     for label in ylabels:
#         for col in df.columns:
#             mnemonic = get_mnemonic(col, alias_dict=alias_dict)
#             if (mnemonic==label) and (mnemonic not in ylabels_alias):            
#                ylabels_alias.append(col)

#     # double check all mnemonic curve data are in df    
#     if (len(Xlables)==len(xlabels_alias)) and (len(ylabels)==len(ylabels_alias)):

#         # dropna if present
#         df.dropna(axis=0, inplace=True)
#         if X is None:
#             X = df[xlabels_alias].values.reshape(-1,len(Xlables))
#         else:
#             X = np.r_[X, df[xlabels_alias].values.reshape(-1,len(Xlables))]
        
#         if y is None:
#             y = df[ylabels_alias].values.reshape(-1,len(ylabels))
#         else:
#             y = np.r_[y, df[ylabels_alias].values.reshape(-1,len(ylabels))]



# print('Data X and y shape:', X.shape, y.shape)

# models = {
#         'RCV': RCV(), 
#         'LSVR': LSVR(epsilon=0.1),
#         'KNN': KNN(n_neighbors=10),
#         'GBR': GBR(),
#         'MLP': MLP(hidden_layer_sizes=(10,)),
#         'XGB': XGB(tree_method='hist', objective='reg:squarederror', n_estimators=100),
# }

# for name, model in models.items():

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)    
#     scaler_x, scaler_y = RobustScaler(), RobustScaler()
#     X_train = scaler_x.fit_transform(X_train)
#     y_train = scaler_y.fit_transform(y_train)

#     reg = model.fit(X_train, y_train)

#     X_test = scaler_x.transform(X_test)
#     y_predict = scaler_y.inverse_transform(reg.predict(X_test).reshape(-1,1))   
    
#     # not the correct way, should scale train and test data seprately
#     X_ = scaler_x.fit_transform(X)
#     y_ = scaler_x.fit_transform(y)
#     text = f'{name} - rmse_cv: {CV_weighted(reg, X_, y_)}'
    
#     # plot crossplot
#     plot_crossplot(y_actual=y_test, 
#                 y_predict=y_predict,
#                 text=text,
#                 plot_show=True,
#                 plot_return=False,
#                 plot_save_file_name=f'Prediction-{name}',
#                 plot_save_path='plots/DTCO-DTSM',
#                 plot_save_format=['png'])
   

#%% create model with desired curves

# create ['DTCO'] as X and ['DTSM'] as y
Xlables = ['DTCO', 'NPHI', 'GR', 'CALI', 'RT']
ylabels = ['DTSM']
X = None
y = None

for key, df in las_dict.items():
    xlabels_alias = []
    ylabels_alias = []

    for label in Xlables:
        for col in df.columns:
            mnemonic = get_mnemonic(col, alias_dict=alias_dict)
            if (mnemonic==label) and (mnemonic not in xlabels_alias):            
               xlabels_alias.append(col)

    for label in ylabels:
        for col in df.columns:
            mnemonic = get_mnemonic(col, alias_dict=alias_dict)
            if (mnemonic==label) and (mnemonic not in ylabels_alias):            
               ylabels_alias.append(col)

    # double check all desired mnemonic curve data are in df    
    if (len(Xlables)==len(xlabels_alias)) and (len(ylabels)==len(ylabels_alias)):

        # check if entire column is na
        for col in df.columns:
            if df[col].isnull().all():
                print(f"Warning! {key}[{col}] is all nan! It's still included in dataset")
        # dropna if present
        df.dropna(axis=0, inplace=True)
        if X is None:
            X = df[xlabels_alias].values.reshape(-1,len(Xlables))
        else:
            X = np.r_[X, df[xlabels_alias].values.reshape(-1,len(Xlables))]
        
        if y is None:
            y = df[ylabels_alias].values.reshape(-1,len(ylabels))
        else:
            y = np.r_[y, df[ylabels_alias].values.reshape(-1,len(ylabels))]



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
    text = f'{name} - rmse_cv: {CV_weighted(reg, X_, y_)}'
    
    # plot crossplot
    plot_crossplot(y_actual=y_test, 
                y_predict=y_predict,
                text=text,
                plot_show=True,
                plot_return=False,
                plot_save_file_name=f'Prediction-{name}',
                plot_save_path='plots/DTCO-DTSM',
                plot_save_format=['png'])
   
#%% plot all las

if __name__ == '__main__':        

    with open('data/las_data_DTSM.pickle', 'rb') as f:
        las_dict = pickle.load(f)

    # plot some random las
    key = random.choice(list(las_dict.keys()))
    
    if '001' in key:
        plot_logs_columns(las_dict[key], 
                            well_name=key,
                            plot_show=True)
                            # plot_return=False,
                            # plot_save_file_name=key_,
                            # plot_save_path='plots',
                            # plot_save_format=['png', 'html'])

    # # plot all las
    # for key in las_dict.keys():
        
    #     plot_logs_columns(las_dict[key], 
    #                     well_name=key,
    #                     plot_show=False,
    #                     plot_return=False,
    #                     plot_save_file_name=key_,
    #                     plot_save_path='plots',
    #                     plot_save_format=['png', 'html'])

#%% test get_mnemonics_from_df

if __name__ == '__main__':

    # load las_data_DTSM
    with open('data/las_data.pickle', 'rb') as f:
        las_data = pickle.load(f)

    # load las_data_DTSM
    with open('data/las_data_DTSM.pickle', 'rb') as f:
        las_data_DTSM = pickle.load(f)

    key = '001-00a60e5cc262_TGS'
    df = las_data[key]
    df.head()

    xlabels_alias = []
    ylabels_alias = []

    target_mnemonics = ['DTCO', 'NPHI', 'GR', 'CALI', 'RT', 'DTSM']

        
    df2 = get_df_from_mnemonics(df, target_mnemonics=target_mnemonics)


    plot_logs_columns(df, 
                    plot_show=True, 
                    DTSM_only=True,
                    plot_return=False,
                    plot_save_file_name=f'{key}-rawdata',
                    plot_save_path='plot_demo',
                    plot_save_format=['png', 'html'])
                    
    plot_logs_columns(df2, 
                    plot_show=True, 
                    DTSM_only=True,
                    plot_return=False,
                    plot_save_file_name=f'{key}-cleaneddata',
                    plot_save_path='plot_demo',
                    plot_save_format=['png', 'html'])