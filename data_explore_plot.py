#%% import
import pickle
import numpy as np
import pandas as pd
import lasio
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


from plot import plot_logs_columns, plot_crossplot, plot_outliers
from sklearn.neighbors import LocalOutlierFactor
from itertools import product
from sklearn.linear_model import LinearRegression as MLR
from sklearn.linear_model import HuberRegressor as Huber

pio.renderers.default = "browser"

from load_pickle import alias_dict, las_data_DTSM_QC, test_list
from plot import plot_outliers, plot_logs_columns
from util import get_mnemonic, get_alias, read_las, process_las, read_pkl, to_pkl

test_list = test_list["Test LAS"].tolist()
#%% detect and display outliers

from sklearn.covariance import EllipticEnvelope


def detect_display_outliers(
    df=None, display=False, path=None, las_name=None, alias_dict=None
):

    outlier_detector = EllipticEnvelope(contamination=0.01)
    try:
        labels = outlier_detector.fit_predict(df[["DTCO", "DTSM"]])
    except:
        labels = outlier_detector.fit_predict(df)

    if display:

        df_inliners = df[labels == 1]
        df_outliers = df[["DEPTH", "DTSM"]][labels == -1].copy()
        df_outliers.columns = ["Depth", "DTSM_Pred"]

        plot_logs_columns(
            df=df_inliners,
            DTSM_pred=df_outliers,
            well_name=las_name,
            plot_show=False,
            plot_save_file_name=las_name,
            plot_save_format=["png"],
            plot_save_path=path,
            alias_dict=alias_dict,
        )

    return labels


for las_name, df in las_data_DTSM_QC.items():
    mnemonics_3 = [
        "DEPTH",
        "DTCO",
        "DTSM",
    ]

    df = process_las().get_df_by_mnemonics(
        df=df,
        target_mnemonics=mnemonics_3,
        alias_dict=alias_dict,
        strict_input_output=False,
    )
    detect_display_outliers(
        df=df,
        display=True,
        path="plots/outliers",
        las_name=las_name,
        alias_dict=alias_dict,
    )

#%% remove outliers like well 001


df = las_data_DTSM_QC["001-00a60e5cc262_TGS"].copy()
mnemonics_7 = [
    "DEPTH",
    "DTCO",
    "RHOB",
    "NPHI",
    "GR",
    "RT",
    "CALI",
    "PEFZ",
    "DTSM",
]

df = process_las().get_df_by_mnemonics(
    df=df,
    target_mnemonics=mnemonics_7,
    alias_dict=alias_dict,
    strict_input_output=False,
)
df

plot_logs_columns(df, alias_dict=alias_dict, plot_show=True)

X = df["DTCO"].values.reshape(-1, 1)
y_true = df["DTSM"].values.reshape(-1, 1)
reg = MLR()
reg.fit(X, y_true)
y_pred = reg.predict(X).reshape(-1, 1)

# plot_crossplot(y_actual=y_true, y_predict=y_pred, include_diagnal_line=True)

#%% oneclass SVM
from plot import plot_logs_columns, plot_crossplot, plot_outliers
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.covariance import EllipticEnvelope
from sklearn.neighbors import LocalOutlierFactor
from sklearn.linear_model import HuberRegressor

# outlier_detector = OneClassSVM(kernel="linear", nu=0.3)
# outlier_detector = IsolationForest(contamination=0.05)
outlier_detector = EllipticEnvelope(contamination=0.03)
# outlier_detector = LocalOutlierFactor(n_neighbors=10)

labels = outlier_detector.fit_predict(df[["DTCO", "DTSM"]])

y = np.c_[y_true, y_pred, labels]
inliners = y[y[:, -1] == 1]
outliers = y[y[:, -1] == -1]

# plot_crossplot(
#     y_actual=inliners[:, 0], y_predict=inliners[:, 1], include_diagnal_line=True
# )
# plot_crossplot(
#     y_actual=outliers[:, 0], y_predict=outliers[:, 1], include_diagnal_line=True
# )

df_outliers = df[["DTSM"]][y[:, -1] == -1].copy()
df_outliers["Depth"] = df_outliers.index
df_outliers.columns = ["DTSM_Pred", "Depth"]

plot_logs_columns(
    df=df[y[:, -1] == 1],
    DTSM_pred=df_outliers,
    alias_dict=alias_dict,
    plot_show=True,
)
# plot_logs_columns(df[y[:, -1] == -1], alias_dict=alias_dict, plot_show=True)

#%% remove outlier through regression
from plot import plot_outliers
from sklearn.linear_model import HuberRegressor, LinearRegression

Xy = pd.DataFrame(
    np.c_[X.reshape(-1, 1), y_true.reshape(-1, 1)], columns=["DTCO", "DTSM"]
)

estimator = HuberRegressor()  # estimator = LinearRegression()
estimator.fit(X, y_true)

x1 = np.arange(300).reshape(-1, 1)
y1 = estimator.predict(x1)
abline = pd.DataFrame(np.c_[x1, y1], columns=["DTCO", "DTSM"])

Xy_out = Xy[labels == -1]

plot_outliers(Xy=Xy, Xy_out=Xy_out, abline=abline, axis_range=200, plot_show=True)


#%% TEST lasio, used log paser from here: https://lasio.readthedocs.io/en/latest

# las = lasio.read("data/las/02571837c35f_TGS.las")
# # check existing curves and data shapes
# print(las.curves)
# print(las.well)
# print(las.data.shape)
# read first las file
# las = read_las("data/las/095b70877102_TGS.las")
# df = las.df()

# print(df)

# plot_logs_columns(df)

#%%
df_target_mnemonics = dict()
df_target_mnemonics_avg = dict()
df_target_mnemonics_count = dict()
df_target_mnemonics_count2 = dict()
las2qc = dict()

target_mnemonics = ["DTCO", "RHOB", "NPHI", "GR", "CALI", "RT", "PEFZ"]
target_mnemonics = target_mnemonics + ["DTSM"]

for key in las_data_DTSM_QC.keys():

    df = las_data_DTSM_QC[key]
    print("processing:\t", key)

    for m in target_mnemonics:

        alias = [i for i in get_alias(m) if i in df.columns]
        alias_count = len(alias)
        if m not in df_target_mnemonics_count.keys():
            df_target_mnemonics_count[m] = [[key[:3], alias_count]]
        else:
            df_target_mnemonics_count[m].append([key[:3], alias_count])

        if alias_count >= 2:
            if key not in df_target_mnemonics_count2.keys():
                df_target_mnemonics_count2[key] = [alias]
            else:
                df_target_mnemonics_count2[key] += [alias]

        for col in df.columns:
            if get_mnemonic(col, alias_dict=alias_dict) == m:
                arr = df[col].values.reshape(-1, 1)
                arr_avg = np.mean(arr)
                if m not in df_target_mnemonics.keys():
                    df_target_mnemonics[m] = arr
                    df_target_mnemonics_avg[m] = arr_avg

                else:
                    df_target_mnemonics[m] = np.r_[df_target_mnemonics[m], arr]
                    df_target_mnemonics_avg[m] = np.r_[
                        df_target_mnemonics_avg[m], arr_avg
                    ]

                # check any caliper average smaller than 7
                if (arr_avg < 7 or arr_avg > 12) and m == "CALI":
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any DTCO average more than 100
                if arr_avg >= 100 and m == "DTCO":
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any DTCO average more than 100
                if arr_avg >= 180 and m == "DTSM":
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any NPHI higher than .25
                if arr_avg >= 0.25 and m == "NPHI":
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any RT higher than 2000
                if arr_avg >= 2000 and m == "RT":
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)


#%% plot histogram for each target mnemonic

for key in df_target_mnemonics.keys():
    df_target_mnemonics[key] = pd.DataFrame(df_target_mnemonics[key], columns=[key])
    fig = px.histogram(df_target_mnemonics[key], x=key, title=key)
    fig.update_layout(
        showlegend=True,
        title=dict(text=key, font=dict(size=20)),
        font=dict(size=18),
        template="plotly",
        width=3000,
        height=1200,
    )
    fig.write_image(f"plots/plots_histogram/histogram_{key}.png")
    print(df_target_mnemonics[key].shape)

    df_target_mnemonics_avg[key] = pd.DataFrame(
        df_target_mnemonics_avg[key], columns=[key]
    )
    fig2 = px.histogram(df_target_mnemonics_avg[key], x=key, title=f"{key}-Average")
    fig2.update_layout(
        showlegend=True,
        title=dict(text=f"{key}-Average", font=dict(size=20)),
        font=dict(size=18),
        template="plotly",
        width=3000,
        height=1200,
    )
    fig2.write_image(f"plots/plots_histogram/histogram_{key}_avg.png")

    df_target_mnemonics_count[key] = pd.DataFrame(
        np.array(df_target_mnemonics_count[key]), columns=["las#", key]
    )
    fig3 = px.histogram(df_target_mnemonics_count[key], x=key, title=f"{key}-Count")
    fig3.update_layout(
        showlegend=True,
        title=dict(text=f"{key}-Count", font=dict(size=20)),
        font=dict(size=18),
        template="plotly",
        width=3000,
        height=1200,
    )
    fig3.write_image(f"plots/plots_histogram/histogram_{key}_count.png")

df_target_mnemonics_count
len(df_target_mnemonics_count2)


#%% plot crossplot

mnemonics_x = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]
mnemonics_y = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]


target_mnemonics_ = [[x, y] for x in mnemonics_x for y in mnemonics_y if x != y]

for target_mnemonics in target_mnemonics_:
    las_dict = dict()
    for key in las_data_DTSM_QC.keys():

        df = las_data_DTSM_QC[key]
        print("processing:\t", key)

        df = process_las().despike(df, window_size=5)

        df = process_las().get_df_by_mnemonics(
            df=df,
            target_mnemonics=target_mnemonics,
            strict_input_output=True,
            alias_dict=alias_dict,
        )

        if (df is not None) and len(df > 1):
            las_dict[key] = df

    df_xp = pd.concat([las_dict[k] for k in las_dict.keys()], axis=0)

    plot_crossplot(
        y_actual=df_xp[target_mnemonics[1]].values,
        y_predict=df_xp[target_mnemonics[0]].values,
        text=target_mnemonics,
        plot_show=True,
        plot_return=False,
        plot_save_file_name=f"{target_mnemonics}-Crossplot",
        plot_save_path=f"plots/crossplot",
        plot_save_format=["png", "html"],
    )


#%% plot feature dependence

target_mnemonics_2 = ["RT", "NPHI"]
target_mnemonics = target_mnemonics_2 + ["DTSM"]  # 'DTSM' is a response variable

Xy = process_las().get_compiled_df_from_las_dict(
    las_data_dict=las_data_DTSM_QC,
    target_mnemonics=target_mnemonics,
    alias_dict=alias_dict,
    strict_input_output=True,
    add_DEPTH_col=False,
    log_RT=True,
)

X_train = Xy.iloc[:, :-1]
y_train = Xy.iloc[:, -1:]

# fit the model
from models.models import model_2

model = model_2["XGB"]
model.fit(X_train, y_train)

DTCO_max, NPHI_max = X_train.quantile(q=0.99)
DTCO_min, NPHI_min = X_train.quantile(q=0.01)

x = np.linspace(DTCO_min, DTCO_max, num=50)
y = np.linspace(NPHI_min, NPHI_max, num=50)
xy = product(x, y)

z = []
for (x_, y_) in xy:
    df = pd.DataFrame([[x_, y_]], columns=target_mnemonics_2)
    z.append(list(model_2["XGB"].predict(df))[0])
zz = np.array(z)

xx, yy = np.meshgrid(x, y)
xx = xx.ravel()
yy = yy.ravel()

print(Xy.describe())
#%%
fig = go.Figure(
    data=[
        go.Mesh3d(
            x=xx,
            y=yy,
            z=zz,
            opacity=0.90,
            color="orange",
            colorscale=[[0, "gold"], [0.5, "mediumturquoise"], [1, "magenta"]],
        )
    ]
)

title = "DTSM partial dependence on DCTO and NPHI"
fig.update_layout(
    scene_camera=dict(eye=dict(x=2, y=-2, z=1.0)),
    template="plotly_dark",
    height=1300,
    width=1300,
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    title=dict(text=title, x=0.5, xanchor="center", font=dict(color="Lime", size=20)),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=0.06,
        xanchor="center",
        x=0.5,
    ),
)

fig.update_scenes(
    xaxis=dict(
        title=target_mnemonics_2[0],
        showgrid=False,
        showline=False,
        showbackground=False,
        showticklabels=True,
        # range=[ ]
    ),
    yaxis=dict(
        title=target_mnemonics_2[1],
        showgrid=False,
        showline=False,
        showbackground=False,
        showticklabels=True,
        # range = [ ]
    ),
    zaxis=dict(
        title="DTSM",
        showgrid=False,
        showline=False,
        showbackground=False,
        showticklabels=True,
        # range=(25000, 0)
    ),
),

fig.show()
fig.write_html(
    f"plots/plots_dependence/dependence_{target_mnemonics_2[0]}_{target_mnemonics_2[1]}_DTSM.html"
)
fig.write_image(
    f"plots/plots_dependence/dependence_{target_mnemonics_2[0]}_{target_mnemonics_2[1]}_DTSM.png"
)

#%% plot TEST well relative location in 3D
from load_pickle import las_depth, las_depth_TEST, las_lat_lon, lat_lon_TEST
from plot import plot_wells_3D

for WellTEST in las_depth_TEST.keys():
    plot_wells_3D(
        las_name_test=WellTEST,
        las_depth=dict(**las_depth, **las_depth_TEST),
        las_lat_lon=dict(**las_lat_lon, **lat_lon_TEST),
        num_of_neighbors=1,
        vertical_anisotropy=0.1,
        depth_range_weight=0.1,
        title=WellTEST,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=WellTEST,
        plot_save_path="plots/TEST",
        plot_save_format=["html", "png"],
    )


#%% plot TEST well 2 relative location in 3D

from load_pickle import las_depth, las_lat_lon
from util import read_pkl
from plot import plot_wells_3D

path = "data/leaderboard_3"

las_depth_TEST = read_pkl(f"{path}/las_depth_TEST.pickle")
las_lat_lon_TEST = read_pkl(f"{path}/las_lat_lon_TEST.pickle")

for WellTEST in las_depth_TEST.keys():
    plot_wells_3D(
        las_name_test=WellTEST,
        las_depth=dict(**las_depth, **las_depth_TEST),
        las_lat_lon=dict(**las_lat_lon, **las_lat_lon_TEST),
        num_of_neighbors=1,
        vertical_anisotropy=0.1,
        depth_range_weight=0.1,
        title=WellTEST,
        plot_show=False,
        plot_return=False,
        plot_save_file_name=WellTEST,
        plot_save_path="plots/TEST_3",
        plot_save_format=["html", "png"],
    )

#%% plot different zones across 15k

las_name = "191-cbca0681ee9e_TGS"
df_ = las_data_DTSM_QC[las_name]
path = "data/leaderboard_3"

for yrange in ([0, 1000], [1000, 4000], [4000, 10000]):
    df = df_[(df_.index > float(yrange[0])) & (df_.index <= float(yrange[1]))]
    plot_logs_columns(
        df=df,
        DTSM_pred=None,
        well_name=None,
        alias_dict=alias_dict,
        plot_show=True,
        plot_return=False,
        plot_save_file_name=f"{las_name}-Prediction-Depth",
        plot_save_path=path,
        plot_save_format=None,
    )

    Xy_test = process_las().get_df_by_mnemonics(
        df=df,
        target_mnemonics=["DTCO", "DTSM"],
        log_mnemonics=["RT"],
        strict_input_output=True,
        alias_dict=alias_dict,
        outliers_contamination=None,  # should not remove outliers when testing!!!
        drop_na=True,  # drop_na should be False when predicting 20 test files
    )

    X_test, y_test = Xy_test.values[:, :1], Xy_test.values[:, -1:]
    model = Huber()
    model.fit(X_test, y_test)
    y_pred = model.predict(X_test)

    # plot DTSM vs DTCO
    plot_outliers(
        Xy=Xy_test,
        Xy_out=None,
        abline=pd.DataFrame(np.c_[X_test, y_pred], columns=["DTCO", "DTSM"]),
        text="DTCO vs DTSM",
        axis_range=300,
        plot_show=True,
        plot_return=False,
        plot_save_file_name=f"{las_name}-DVCO-vs-DTSM-Crossplot",
        plot_save_path=None,
        plot_save_format=None,
    )

#%% deal with NPHI less than 0, shift up

las_name = "008-07bda346ac4d_TGS"
df = las_data_DTSM_QC[las_name].copy()
df.columns = df.columns.map(alias_dict)
df

plot_logs_columns(
    df=df,
    DTSM_pred=None,
    well_name=las_name,
    plot_show=True,
    plot_save_file_name=las_name,
    plot_save_format=None,
    plot_save_path=None,
    alias_dict=alias_dict,
)

adj = df[df["NPHI"] < 0]["NPHI"].quantile(q=0.001)
df["NPHI"] = df["NPHI"] - adj


#%% plot pairplot per las
path = "plots/DTCO_DTSM_pairplot"

corr = []
# las_name = "233-fe8ab5538224_TGS"

target_mnemonics = ["DEPTH", "DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ", "DTSM"]
for key, df in las_data_DTSM_QC.items():
    if key in test_list:
        print(key)
        df = process_las().get_df_by_mnemonics(
            df=df.copy(),
            target_mnemonics=target_mnemonics,
            log_mnemonics=["RT"],
            strict_input_output=False,
            alias_dict=alias_dict,
            outliers_contamination=None,  # should not remove outliers when testing!!!
            drop_na=True,  # drop_na should be False when predicting 20 test files
        )

        x = df[["DTCO"]].copy()
        y = df[["DTSM"]].copy()
        reg = Huber()
        reg.fit(x, y)
        coef = reg.coef_
        intercept = reg.intercept_
        corr.append([key, coef[0], intercept])

        y_ = reg.predict(x)
        abline = pd.DataFrame(
            np.c_[x.values.reshape(-1, 1), y_.reshape(-1, 1)], columns=["DTCO", "DTSM"]
        )

        plot_outliers(
            Xy=df,
            Xy_out=None,
            abline=abline,
            text=f"Coef:{coef}",
            axis_range=300,
            plot_show=False,
            plot_return=False,
            plot_save_file_name=f"{key}-Huber",
            plot_save_path=path,
            plot_save_format=["png"],
        )

        # plot pred vs true DTSM
        plot_logs_columns(
            df=df,
            DTSM_pred=None,
            well_name=key,
            alias_dict=alias_dict,
            plot_show=False,
            plot_return=False,
            plot_save_file_name=f"{key}-logs",
            plot_save_path=path,
            plot_save_format=["png"],
        )

corr = pd.DataFrame(corr, columns=["WellName", "coef", "intercept"])
corr.to_csv("plots/DTCO_DTSM_pairplot/coef.csv")

#%% pairplot for individual well

key = "064-4e536dc14f41_TGS"
df = process_las().get_df_by_mnemonics(
    df=las_data_DTSM_QC["064-4e536dc14f41_TGS"].copy(),
    target_mnemonics=target_mnemonics,
    log_mnemonics=["RT"],
    strict_input_output=False,
    alias_dict=alias_dict,
    outliers_contamination=None,  # should not remove outliers when testing!!!
    drop_na=True,  # drop_na should be False when predicting 20 test files
)

Xy = df.copy()
fig = px.scatter(Xy, x="DTCO", y="DTSM", color="RHOB")

fig.update_layout(
    title=dict(text=key),
    width=1200,
    height=1200,
    xaxis_range=[40, 120],  # "DTCO": [40, 120],
    yaxis_range=[50, 250],  # "DTSM": [60, 270],
)


#%% plot pairplot per for all data
fig.write_image("plots/DTCO_DTSM_pairplot/DTCO_DTSM_RHOB.png")

from itertools import product

# las_name = "233-fe8ab5538224_TGS"
a = ["DTCO", "RHOB", "NPHI", "GR", "PEFZ"]
b = ["DTSM"]
target_mnemonics_ = product(a, b)

for target_mnemonics in target_mnemonics_:
    print(target_mnemonics)
    target_mnemonics = list(target_mnemonics)
    Xy = process_las().get_compiled_df_from_las_dict(
        las_data_dict=las_data_DTSM_QC,
        target_mnemonics=target_mnemonics,
        log_mnemonics=["RT"],
        strict_input_output=True,
        outliers_contamination=None,
        alias_dict=alias_dict,
        return_dict=False,
        drop_na=True,
    )

    print(Xy.head())
    # fig = px.scatter_matrix(Xy)
    fig = px.scatter(Xy, x=target_mnemonics[0], y=target_mnemonics[1])
    fig.update_layout(title_text=str(target_mnemonics), width=1300, height=1300)
    fig.write_image(f"plots/scatter_matrix_alldata/{target_mnemonics}.png")
    # fig.write_html(f"plots/scatter_matrix_alldata/{target_mnemonics}.html")


#%% outlier for DTCO vs DTSM

path = "data/feature_selected_data"
df1 = read_pkl(f"{path}/las_dict_1_2.pickle")
df1 = pd.concat([i for i in df1.values()])

from sklearn.svm import OneClassSVM

outlier_detector = OneClassSVM(kernel="linear")
labels = outlier_detector.fit_predict(df1)
df2 = df1[labels == 1]
df3 = df1[labels == -1]


import plotly.express as px

px.scatter(df1, x="DTCO", y="DTSM")
px.scatter(df2, x="DTCO", y="DTSM")
px.scatter(df3, x="DTCO", y="DTSM")

#%% felix data

well_names = dict()

for key in las_data_DTSM_QC.keys():
    well_names[key.split("-")[1]] = key


df1 = pd.read_csv("data/felix_data/names.csv")
df1.head()
df1["File_new"] = df1["File"].map(well_names)

well_names_ = dict()
for _, well, _, file_new in df1.itertuples():
    well_names_[well] = file_new

df = pd.read_csv("data/felix_data/train.csv")
df.head()
df.columns = ["DEPTH", "GR", "CALI", "NPHI", "RHOB", "PEFZ", "RT", "DTCO", "DTSM", "ID"]

df.describe()
df["RT"]
ids = np.unique(df["ID"])

las_data_DTSM_felix = dict()

for id in ids:
    id_ = well_names_[id]
    las_data_DTSM_felix[id_] = df[df.columns.difference(["ID"])][df["ID"] == id]

to_pkl(las_data_DTSM_felix, f"data/felix_data/las_data_DTSM_felix.pickle")
