#%% this module is to used test different models
import os
import pickle
import time

import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import xgboost
from scipy.spatial.distance import cdist
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from xgboost import XGBRegressor as XGB

from load_pickle import alias_dict, las_data_DTSM_QC
from plot import plot_crossplot, plot_logs_columns

pio.renderers.default = "browser"

# load customized functions and requried dataset
from util import add_gradient_features, process_las, read_pkl, to_pkl


#%% clustering to identify zones

path = "predictions/tuning/KMeans_zoning_DTCO"
if not os.path.exists(path):
    os.mkdir(path)

# model_name = "zoning_DTCO_DTSM"
# target_mnemonics = ["DTCO", "DTSM"]

model_name = "zoning_DTCO"
target_mnemonics = ["DTCO", "DTSM"]  # , "GR"]


try:
    las_dict = read_pkl(f"{path}/las_dict_{model_name}.pickle")
except:
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

#%% prepare the data

X_raw = []
for key, df in las_dict.items():

    # add gradients as features
    # df = add_gradient_features(df)
    X_raw.append(df)

X_raw = pd.concat(X_raw, axis=0)

X_raw.describe()

# # scale the data
# scaler_x = RobustScaler()
# X = scaler_x.fit_transform(X_raw.values)
#%% use DTCO vs DTSM slope and intercept as input data for clustering
dt = pd.read_csv(f"data/DTCO_DTSM_corr.csv")
X_raw = dt[["coef", "intercept"]]
scaler_x = RobustScaler()
X = scaler_x.fit_transform(X_raw)


n_clusters = 4
# Building and fitting the model
kmeanModel = KMeans(n_clusters=n_clusters)
kmeanModel.fit(X)

KMeans_model = dict(
    scaler_x=scaler_x, model=kmeanModel, target_mnemonics=target_mnemonics
)  #

to_pkl(
    KMeans_model, f"{path}/KMeans_model_[DTCO_DTSM_corr]_{n_clusters}_clusters.pickle"
)

X_pca = X_raw.copy()
X_pca["color"] = kmeanModel.labels_

# X_pca_ = X_pca.sample(100000)
# fig = px.scatter_3d(X_pca_, x="DTCO", y="NPHI", z="GR", color="color")
fig = px.scatter(X_pca, x="coef", y="intercept", color="color")

fig.update_layout(
    width=2400,
    height=1200,
    title_text=f"{n_clusters} Clusters using KMeans with [DTCO]",
)
fig.show()
# fig.write_html(f"{path}/{n_clusters}_clusters_[DTCO].html")
# fig.write_image(f"{path}/{n_clusters}_clusters_[DTCO,NPHI,GR].png")

#%% KMeans elbow method, no need to run everytime
# X = X_raw[["DTCO"]]

distortions = []
inertias = []

for k in range(1, 10):
    time0 = time.time()

    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)

    distortions.append(
        [
            k,
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0],
        ]
    )
    inertias.append([k, kmeanModel.inertia_])

    print(f"KMeans with {k} clusters in {time.time()-time0:.1f} s")

print("Completed clustering!")


#% plot the elbow plots
distortions = pd.DataFrame(distortions, columns=["k_clusters", "distortion"])
inertias = pd.DataFrame(inertias, columns=["k_clusters", "inertia"])

fig = px.line(distortions, x="k_clusters", y="distortion")
fig.data[0].update(mode="lines+markers")
fig.update_layout(title_text=f"{model_name}_distortions", width=2400, height=1200)
fig.write_image(f"{path}/{model_name}_elbow_distortions.png")

fig = px.line(inertias, x="k_clusters", y="inertia")
fig.data[0].update(mode="lines+markers")
fig.update_layout(title_text=f"{model_name}_inertias", width=2400, height=1200)
fig.write_image(f"{path}/{model_name}_elbow_inertias.png")

# seems 3 clusters is the optimal number
#%% [DTCO,NPHI,GR]

n_clusters = 2
# Building and fitting the model
kmeanModel = KMeans(n_clusters=n_clusters)
kmeanModel.fit(X)

KMeans_model = dict(
    model=kmeanModel, target_mnemonics=target_mnemonics
)  # scaler_x=scaler_x,

to_pkl(KMeans_model, f"{path}/KMeans_model_[DTCO]_{n_clusters}_clusters.pickle")

X_pca = X_raw.copy()
X_pca["color"] = kmeanModel.labels_

X_pca_ = X_pca.sample(100000)
# fig = px.scatter_3d(X_pca_, x="DTCO", y="NPHI", z="GR", color="color")
fig = px.scatter(X_pca_, x="DTCO", y="DTSM", color="color")

fig.update_layout(
    width=2400,
    height=1200,
    title_text=f"{n_clusters} Clusters using KMeans with [DTCO]",
)
fig.show()
fig.write_html(f"{path}/{n_clusters}_clusters_[DTCO].html")
# fig.write_image(f"{path}/{n_clusters}_clusters_[DTCO,NPHI,GR].png")

#%% DTCO vs DTSM

# n_clusters = 4
# # Building and fitting the model
# kmeanModel = KMeans(n_clusters=n_clusters)
# kmeanModel.fit(X)

# KMeans_model = dict(
#     scaler_x=scaler_x,
#     model=kmeanModel,
# )

# X_pca = X_raw.copy()
# X_pca["color"] = kmeanModel.labels_

# X_pca_ = X_pca.sample(500000)
# fig = px.scatter(X_pca_, x="DTCO", y="DTSM", color="color")
# fig.update_layout(width=2400, height=1200, title_text="4 Clusters using KMeans")
# fig.show()
# fig.write_html(f"{path}/clusters_DTCO_DTSM.html")
# fig.write_image(f"{path}/clusters_DTCO_DTSM.png")

#%% plot zones for each las

# # las_name = "001-00a60e5cc262_TGS"
# # df = las_data_DTSM_QC[las_name]

# path = "plots/5zones"
# KMeans_model = read_pkl(f"{path}/KMeans_model_[DTCO,NPHI,GR]_5_clusters.pickle")

# # target_mnemonics = ["DTCO", "NPHI", "GR"]
# # KMeans_model["target_mnemonics"] = ["DTCO", "NPHI", "GR"]
# # to_pkl(KMeans_model, f"{path}/KMeans_model_[DTCO,NPHI,GR]_5_clusters.pickle")

# for las_name, df in las_data_DTSM_QC.items():

#     df = process_las().get_df_by_mnemonics(
#         df=df,
#         target_mnemonics=KMeans_model["target_mnemonics"],
#         log_mnemonics=["RT"],
#         strict_input_output=True,
#         alias_dict=alias_dict,
#         drop_na=True,
#     )

#     if df is not None and len(df) > 10:

#         df["labels"] = KMeans_model["model"].predict(
#             KMeans_model["scaler_x"].transform(df)
#         )

#         plot_logs_columns(
#             df=df,
#             DTSM_pred=None,
#             well_name=las_name,
#             alias_dict=alias_dict,
#             plot_show=False,
#             plot_return=False,
#             plot_save_file_name=f"KMeans_clustering_{las_name}",
#             plot_save_path=path,
#             plot_save_format=["png"],
#         )


#%% check std deviation of logs
# '022-16bb5ea9d2d3_TGS', '023-1737d6c90d5e_TGS','109-84967b1f42e0_TGS',
df1 = las_data_DTSM_QC["022-16bb5ea9d2d3_TGS"]
df2 = las_data_DTSM_QC["023-1737d6c90d5e_TGS"]
df3 = las_data_DTSM_QC["109-84967b1f42e0_TGS"]
target_mnemonics = ["DTCO", "DTSM"]

df1 = process_las().get_df_by_mnemonics(
    df=df1,
    target_mnemonics=target_mnemonics,
    log_mnemonics=["RT"],
    strict_input_output=True,
    alias_dict=alias_dict,
    outliers_contamination=None,  # should not remove outliers when testing!!!
    drop_na=True,  # drop_na should be False when predicting 20 test files
)
print("df", df1.describe())

df2 = process_las().get_df_by_mnemonics(
    df=df2,
    target_mnemonics=target_mnemonics,
    log_mnemonics=["RT"],
    strict_input_output=True,
    alias_dict=alias_dict,
    outliers_contamination=None,  # should not remove outliers when testing!!!
    drop_na=True,  # drop_na should be False when predicting 20 test files
)
print("df", df2.describe())


df3 = process_las().get_df_by_mnemonics(
    df=df3,
    target_mnemonics=target_mnemonics,
    log_mnemonics=["RT"],
    strict_input_output=True,
    alias_dict=alias_dict,
    outliers_contamination=None,  # should not remove outliers when testing!!!
    drop_na=True,  # drop_na should be False when predicting 20 test files
)
print("df", df3.describe())
x = df3.describe()


X = df1.copy()

X["color"] = kmeanModel.predict(X[["DTCO"]])

fig = px.scatter(X, x="DTCO", y="DTSM", color="color")

fig.update_layout(
    width=2400,
    height=1200,
    title_text=f"{n_clusters} Clusters using KMeans with [DTCO]",
)
fig.show()
# fig.write_html(f"{path}/{n_clusters}_clusters_[DTCO].html")


#%%
x_ix = []
X = []
target_mnemonics = ["DTCO"]

for las_name, df in las_data_DTSM_QC.items():

    df = process_las().get_df_by_mnemonics(
        df=df,
        target_mnemonics=target_mnemonics,
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        outliers_contamination=None,  # should not remove outliers when testing!!!
        drop_na=True,  # drop_na should be False when predicting 20 test files
    )
    if df is not None and len(df) > 100:
        x_ix.append(las_name)
        X.append(df.describe().transpose().values.tolist()[0][1:])
        # X.append(df.describe().transpose().at["RHOB", "std"])

X = pd.DataFrame(X)
X.index = x_ix
print(X)

# from sklearn.preprocessing import StandardScaler

# scaler_x = StandardScaler()
# X = scaler_x.fit_transform(X)

#%%
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import time

distortions = []
inertias = []

model_name = "RHOB"
path = "plots/RHOB_zone"
for k in range(1, 10):
    time0 = time.time()
    # Building and fitting the model
    kmeanModel = KMeans(n_clusters=k)
    kmeanModel.fit(X)

    distortions.append(
        [
            k,
            sum(np.min(cdist(X, kmeanModel.cluster_centers_, "euclidean"), axis=1))
            / X.shape[0],
        ]
    )
    inertias.append([k, kmeanModel.inertia_])

    print(f"KMeans with {k} clusters in {time.time()-time0:.1f} s")

print("Completed clustering!")


#% plot the elbow plots
distortions = pd.DataFrame(distortions, columns=["k_clusters", "distortion"])
inertias = pd.DataFrame(inertias, columns=["k_clusters", "inertia"])

fig = px.line(distortions, x="k_clusters", y="distortion")
fig.data[0].update(mode="lines+markers")
fig.update_layout(title_text=f"{model_name}_distortions", width=2400, height=1200)
fig.write_image(f"{path}/{model_name}_elbow_distortions.png")

fig = px.line(inertias, x="k_clusters", y="inertia")
fig.data[0].update(mode="lines+markers")
fig.update_layout(title_text=f"{model_name}_inertias", width=2400, height=1200)
fig.write_image(f"{path}/{model_name}_elbow_inertias.png")


n_clusters = 2
# Building and fitting the model
kmeanModel = KMeans(n_clusters=n_clusters)
kmeanModel.fit(X)

KMeans_model = dict(
    scaler_x=scaler_x, model=kmeanModel, target_mnemonics=target_mnemonics
)
#% save the cluster model
to_pkl(kmeanModel, "models/KMeans_model_RHOB_std_2_clusters.pickle")

# kmeanModel.predict(df3["RHOB"].std().reshape(-1,1))
kmeanModel.predict(df3.describe().transpose().values[0][1:].reshape(1, -1))
