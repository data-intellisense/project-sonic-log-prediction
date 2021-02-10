#%% import lib
import pickle
import random
import pandas as pd
from plot import plot_logs_columns
from util import get_mnemonic, read_las, process_las, las_name_test, get_distance
from load_pickle import alias_dict, las_data_DTSM_QC

#%% plot wellbores in 3D, weighted and un-weighted

plot_3DWell(
    las_name_test=las_name_test, las_data_DTSM=las_data_DTSM, display_weight=True
)
plot_3DWell(
    las_name_test=las_name_test, las_data_DTSM=las_data_DTSM, display_weight=False
)

#%% -----------------------------------------------------------------------------------------
# plot all las and save the plots

# plot some random las for testing
key = random.choice(list(las_data_DTSM.keys()))

plot_logs_columns(las_data_DTSM[key], well_name=key, plot_show=True)

# plot all las
for key in las_data_DTSM.keys():
    plot_logs_columns(
        las_data_DTSM[key],
        well_name=key,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=key,
        plot_save_path="plots",
        plot_save_format=["png", "html"],
    )

#%% -----------------------------------------------------------------------------------------
# TEST: despike

# las = "data/las/0052442d0162_TGS.las"
las = "data/las/0a65a72dd23f_TGS.las"
df = read_las(las).df()

plot_logs_columns(df)

from scipy.signal import medfilt

medfilt(df["DTCO"].values, kernel_size=3)

df = process_las().despike(df)
plot_logs_columns(df)

#%% -----------------------------------------------------------------------------------------
# create coordinate plot

import pandas as pd
import numpy as np
import seaborn

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

pio.renderers.default = "browser"

# import cordinates
cords = pd.read_csv("data/cords.csv", index_col=0)

print(cords.sample(5))

fig = go.Figure()
fig.add_traces(
    go.Scatter(
        x=cords["Lon"],
        y=cords["Lat"],
        mode="markers",
        marker=dict(size=cords["STOP"] / 500),
        hoverinfo="text",
        hovertext=cords["Well"],
    )
)
fig.update_layout(
    xaxis=dict(title="Longitude"),
    yaxis=dict(title="Latitude"),
    title=dict(text="Size: Stop Depth"),
    font=dict(size=18),
)


#%% -----------------------------------------------------------------------------------------
# TEST: mnemonics mapping
las_path = r"data/las/00a60e5cc262_TGS.las"
df = lasio.read(las_path).df()
print("before mnemonics conversion:", df.columns)

# convert different mnemonics to consistent mnemonic
df.columns = df.columns.map(alias_dict)
print("after mnemonics conversion:", df.columns)


#%% -----------------------------------------------------------------------------------------
# TEST get_df_by_mnemonics

las = "data/las/0052442d0162_TGS.las"
df = read_las(las).df()

print("original df:", df.head(5))
print(
    "\nnew df:",
    process_las().get_df_by_mnemonics(
        df=df, target_mnemonics=["DTCO", "GR", "DTSM"], strict_input_output=False
    ),
)

print(
    "\nnew df:",
    process_las().get_df_by_mnemonics(
        df=df,
        target_mnemonics=["DTCO", "GR", "DPHI", "DTSM"],
        strict_input_output=False,
    ),
)

print(
    "\nnew df:",
    process_las().get_df_by_mnemonics(
        df=df, target_mnemonics=["DTCO", "GR", "DPHI", "DTSM"], strict_input_output=True
    ),
)


#%% -----------------------------------------------------------------------------------------
#  TEST: petropy, not working!!!
# import petropy as ptr
# las_path = r"data/las/00a60e5cc262_TGS.las"
# log = ptr.log_data(las_path)
# viewer_sample = ptr.LogViewer(log,top = 6950, height = 100)


#%% -----------------------------------------------------------------------------------------
#  TEST 1: split train/test within rows of data

# # load las_data_DTSM
# with open('data/las_data_DTSM.pickle', 'rb') as f:
#     las_data_DTSM = pickle.load(f)

# # 7 features
# target_mnemonics = ['DTCO', 'NPHI','RHOB', 'GR', 'CALI', 'RT', 'PEFZ', 'DTSM']
# df = None
# key_list = []
# for key in las_data_DTSM.keys():
#     print(f'Loading {key}')
#     df_ = process_las().get_df_by_mnemonics(las_data_DTSM[key], target_mnemonics=target_mnemonics, strict_input_output=True)
#     if df_ is not None:
#         key_list.append(key)
#         if df is None:
#             df = df_
#         else:
#             df = pd.concat([df, df_], axis=0)
# print(f'Total {len(key_list)} las loaded and total {len(df)} rows of data!')
# with open('data/las_data_7features.pickle', 'wb') as f:
#     pickle.dump(df, f)


# #% fit models with 7 features dataset

# X = df.iloc[:, :-1]
# y = df.iloc[:, -1:]

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
#     time0 = time.time()
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
#     text = f'{name} - rmse_cv: {CV_weighted(reg, X_, y_):.2f}'

#     # plot crossplot
#     plot_crossplot(y_actual=y_test.values,
#                 y_predict=y_predict,
#                 text=text,
#                 plot_show=True,
#                 plot_return=False,
#                 plot_save_file_name=f'Prediction-{name}',
#                 plot_save_path='predictions/7features',
#                 plot_save_format=['png'])

#     print(f'Finished fitting with {name} model in {time.time()-time0:.2f} seconds')

#%% for evaluate models based on rmse

import lasio
from models.blackbox_model import blackbox_model
from sklearn.metrics import mean_squared_error


def evaluate(las):

    df = lasio.read(las).df()
    df["DEPT"] = df.index

    if "DTSM" in df.columns:

        # find the rows where DTSM is not NaN
        index_DTSM = ~df["DTSM"].isnull()

        return (
            mean_squared_error(
                blackbox_model(las).loc[index_DTSM, ["DTSM"]],
                df.loc[index_DTSM, ["DTSM"]].loc[index_DTSM],
            )
            ** 0.5
        )
    else:
        return "No DTSM found in las, no prediction made!"


# the path to las file
las_path = r"data/las/00a60e5cc262_TGS.las"
print(f"The rmse of your DTSM predictions for {las_path} is:", evaluate(las_path))


#%%
import numpy as np
import pandas as pd

a = np.array([[1, 2], [2, 3], [3, 4]])

a0 = a[:, 0:1]
a1 = a[:, 1:2]
w = [0.9, 0.1]
a2 = a0 * w[0] + a1 * w[1]

np.mean(a0, axis=1)
a = pd.DataFrame(a)
a.diff(periods=1)

#%%

from load_pickle import las_data_TEST, las_depth, las_lat_lon

# for key, val in las_data_TEST.items():

#     val.to_csv(f"data/leaderboard_1/{key}_index.csv")

#% test get_nearest_neighbors
from util import get_nearest_neighbors

from plot import plot_wells_3D

# t = "214-e8681aef711d_TGS"
# t = "032-1f901b2ab8a5_TGS"
t = "070-5453b3a29db6_TGS"
a = las_depth[t]
las_lat_lon[t]

get_nearest_neighbors(
    depth_TEST=las_depth[t],
    lat_lon_TEST=las_lat_lon[t],
    las_depth=las_depth,
    las_lat_lon=las_lat_lon,
    num_of_neighbors=20,
    vertical_anisotropy=1,
    depth_range_weight=0.1,
)

# plot_wells_3D(
#     las_name_test=t,
#     las_depth=las_depth,
#     las_lat_lon=las_lat_lon,
#     num_of_neighbors=20,
#     vertical_anisotropy=1,  # lower value means less important on lat-lon
#     depth_range_weight=0.01,  # lower value means less importance on range of data
#     plot_save_path="misc",
#     plot_save_format=["png", "html"],
# )

#%% use difference

# create training data dataframe
# if use_difference:

#     neighbors = get_nearest_neighbors(
#         depth_TEST=las_depth[las_name],
#         lat_lon_TEST=las_lat_lon[las_name],
#         las_depth=las_depth,
#         las_lat_lon=las_lat_lon,
#         num_of_neighbors=30,
#         vertical_anisotropy=1,
#         depth_range_weight=0.1,
#     )

#     las_dict_diff = []

#     for k in neighbors:
#         k = k[0]
#         if k not in [las_name]:
#             print("k:", k)
#             temp = las_dict[k].diff(periods=1, axis=0)
#             temp.iloc[0, :] = temp.iloc[1, :]
#             las_dict_diff.append(temp)

#     Xy_train_diff = pd.concat(las_dict_diff, axis=0, ignore_index=True)
#     X_train_diff = Xy_train_diff.values[:, :-1]
#     y_train_diff = Xy_train_diff.values[:, -1:]

#     # scale train data
#     scaler_x_diff, scaler_y_diff = RobustScaler(), RobustScaler()
#     X_train_diff = scaler_x_diff.fit_transform(X_train_diff)
#     y_train_diff = scaler_y_diff.fit_transform(y_train_diff)

#     print("difference training", len(sample_weight), len(X_train_diff))
#     model_diff = list(dmodels.values())[0]
#     try:
#         model_diff.fit(
#             X_train_diff, y_train_diff, sample_weight=sample_weight
#         )
#     except:
#         model_diff.fit(X_train_diff, y_train_diff)
#         print(
#             "Model_diff does not accept sample weight so sample weight was not used in difference training!"
#         )

#     X_test_diff = Xy_test.iloc[:, :-1].diff(periods=1, axis=0)
#     X_test_diff.iloc[0, :] = X_test_diff.iloc[1, :]
#     X_test_diff = X_test_diff.values

#     X_test_diff = scaler_x_diff.transform(X_test_diff)
#     y_predict_diff = scaler_y_diff.inverse_transform(
#         model_diff.predict(X_test_diff).reshape(-1, 1)
#     )
#     y_predict_diff = np.cumsum(y_predict_diff, axis=0)

#     y_predict = (
#         np.mean(y_predict, axis=0)
#         - np.mean(y_predict_diff, axis=0)
#         + y_predict_diff
#     )