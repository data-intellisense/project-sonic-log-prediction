#%% this module is to test deifferent models

from util import get_features_df

with open('data/las_data_DTSM.pickle', 'rb') as f:
    las_dict = pickle.load(f)

#%% create simple DTSM/DTCO model


