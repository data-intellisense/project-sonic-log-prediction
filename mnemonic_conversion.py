import pandas as pd
import pickle
import lasio
from welly import well
from welly import curve

df = pd.read_csv("data/grouped_mnemonics_corrected.csv")
df.sample(10)

mnemonic_dict = dict()
for ix, m1, m2, _ in df.itertuples():
    mnemonic_dict[m2] = m1

with open("data/mnemonic_dict.pickle", "wb") as f:
    pickle.dump(mnemonic_dict, f)

las_path = r"data/las/00a60e5cc262_TGS.las"
df = lasio.read(las_path).df()
df.head()

# convert different mnemonics to consistent mnemonic
df.columns = df.columns.map(mnemonic_dict)
df.head()  # column names that are NaN probably not a good feature so could be dropped


well_las = well.Well()
well_las.add_curves_from_las(las_path, remap=mnemonic_dict)
well_las