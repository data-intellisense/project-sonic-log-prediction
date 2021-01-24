import pandas as pd
import pickle
import lasio
from welly import well
from welly import curve

df = pd.read_csv("data/grouped_mnemonics_corrected.csv")
df.head(10)

alias_dict = dict()
for ix, m1, m2, _ in df.itertuples():
    alias_dict[m1] = m2

with open("data/alias_dict.pickle", "wb") as f:
    pickle.dump(alias_dict, f)

#%% test
las_path = r"data/las/00a60e5cc262_TGS.las"
df = lasio.read(las_path).df()
print('before mnemonics conversion:', df.columns)

# convert different mnemonics to consistent mnemonic
df.columns = df.columns.map(alias_dict)
print('after mnemonics conversion:', df.columns)

