
#%% load necessary data for main.py
import pickle

with open(f"data/las_data_DTSM_QC.pickle", "rb") as f:
    las_data_DTSM_QC = pickle.load(f)

# get las_lat_lon
with open(f"data/las_lat_lon.pickle", "rb") as f:
    las_lat_lon = pickle.load(f)

with open(f"data/leaderboard_1/las_lat_lon.pickle", "rb") as f:
    lat_lon_TEST = pickle.load(f)

# get the alias_dict, required
with open(f"data/alias_dict.pickle", "rb") as f:
    alias_dict = pickle.load(f)
