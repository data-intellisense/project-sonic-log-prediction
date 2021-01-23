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
print(df.head())

# convert different mnemonics to consistent mnemonic
df.columns = df.columns.map(mnemonic_dict)
print(df.head())  # column names that are NaN probably not a good feature so could be dropped

las_path = r"data/las/0a65a72dd23f_TGS.las"
well_las = well.Well()
well_las.add_curves_from_las(las_path, remap=mnemonic_dict)
well_las.df(keys=['DTCO', 'GRR']) 
# well_las.data_as_matrix(keys=['DTCO', 'GRR']) 

well_las.get_curve('DTCO')
well_las.get_mnemonic('DTCO', alias = mnemonic_dict)
well_las.plot()



# pass the data from curve
x = well_las.get_curve('DTCO').data
well_curve = curve.Curve(x)
well_curve.describe()
well_curve.despike()
well_curve.get_stats()
well_curve.get_alias(alias=mnemonic_dict)
well_curve.mnemonic

def get_alias(mnemonic, alias=None):
    alias = alias or {}
    return [k for k, v in alias.items() if mnemonic in v]

get_alias('DPHI', alias=mnemonic_dict)

def convert_alias(alias_ = None, alias=None):
    alias = alias or {}
    return [(k,v) for k, v in alias.items() if alias_ == k]

convert_alias(alias_ = 'DT', alias=mnemonic_dict)

mnemonic_dict.keys()