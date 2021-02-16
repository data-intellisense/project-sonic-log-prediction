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
