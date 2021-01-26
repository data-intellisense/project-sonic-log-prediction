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
