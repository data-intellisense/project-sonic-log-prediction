import lasio
import pandas as pd

# you model should be a function like below


def blackbox_model(las=None):
    """
    input: path to any las file
    output: a pd.DataFrame with the length of input las, columns = ['DEPT', 'DTSM']
    """
    try:
        df = lasio.read(las).df()

        # lasio reads all las 'DEPT' as index, for convenience, we recreate 'DEPT' column
        df["DEPT"] = df.index
        df = df[["DEPT", "DTSM"]]

        df["DTSM"] = 120  # predict DTSM at all depth to be 120, for testing only
        return df

    except:
        # las should have minimum 'DEPT' and 'DTSM' two curves, or else there is no point
        # training or predictiong 'DTSM' in this case
        print("Missing 'DEPT' and 'DTSM' curves, no predictions were made!")
