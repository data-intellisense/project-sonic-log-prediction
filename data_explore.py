from plot import plot_logs
import pandas as pd
import random

# load the file
las_data = pd.HDFStore("data/las_data.h5")
las_curves = pd.HDFStore("data/las_curves.h5")

#% checking a random file
f_name = random.choice(las_data.keys())
print(las_curves[f_name])
print(las_data[f_name])

#%% plot the logs

df = las_data[f_name]
df_curves = las_curves[f_name]
plot_logs(df, f_name)