
#%% load necessary data for main.py
import pickle

try:
    with open(f"data/las_data_DTSM_QC.pickle", "rb") as f:
        las_data_DTSM_QC = pickle.load(f)
except:
    print(" No 'las_data_DTSM_QC.pickle' loaded as it is NOT available!")
    las_data_DTSM_QC = None

# get las_lat_lon
try:
    with open(f"data/las_lat_lon.pickle", "rb") as f:
        las_lat_lon = pickle.load(f)
except:
    print(" No 'las_lat_lon.pickle' loaded as it is NOT available!")
    las_lat_lon = None


# get las_lat_lon
try:
    with open(f"data/las_depth.pickle", "rb") as f:
        las_depth = pickle.load(f)
except:
    print(" No 'las_depth.pickle' loaded as it is NOT available!")
    las_depth = None


try:
    with open(f"data/leaderboard_1/las_lat_lon.pickle", "rb") as f:
        lat_lon_TEST = pickle.load(f)
except:
    print(" No 'las_lat_lon.pickle' loaded as it is NOT available!")
    lat_lon_TEST = None

# get the alias_dict, required
try:
    with open(f"data/alias_dict.pickle", "rb") as f:
        alias_dict = pickle.load(f)
except:
    print(" No 'alias_dict.pickle' loaded as it is NOT available!")
    alias_dict = None


try:
    with open(f"data/leaderboard_1/las_data_TEST.pickle", "rb") as f:
        las_data_TEST = pickle.load(f)
except:
    print(" No 'las_data_TEST.pickle' loaded as it is NOT available!")
    las_data_TEST = None
