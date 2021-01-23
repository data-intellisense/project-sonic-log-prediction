from plot import plot_logs
import pandas as pd
import random
import numpy as np
import welly

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


#%% clean mnemonic

mnemonics = []
for curve in las_curves.keys():
    for _, c, c_ in las_curves[curve].itertuples():
        if c not in mnemonics:
            mnemonics.append([c, c_])
            print([c, c_])

mnemonics = pd.DataFrame(mnemonics, columns=["mnemonic", "description"])
mnemonics.sample(5)

mnemonics.describe()
mnemonics = mnemonics[mnemonics["mnemonic"] != "DEPT"]  # remove 'DEPT'
mnemonics_unique_list = mnemonics["mnemonic"].unique()

# check 'DTCO' and 'DTSM'
mnemonics[mnemonics["mnemonic"] == "DTCO"]
mnemonics[mnemonics["mnemonic"] == "DTSM"]

# plot histogram
import plotly.express as px

px.histogram(mnemonics, x="mnemonic")

# get the unique mnemonics
mnemonics_unique_df = []
for m in mnemonics_unique_list:
    if m not in mnemonics_unique_df:
        description = (
            mnemonics[mnemonics["mnemonic"] == m].reset_index(drop=True).iloc[0, 1]
        )
        mnemonics_unique_df.append([m, description])

mnemonics_unique_df = pd.DataFrame(
    mnemonics_unique_df, columns=["mnemonic", "description"]
)
print("total unique mnemonics:", len(mnemonics_unique_df))
mnemonics_unique_df.sample(5)

# double check uniqueness of DTCO and DTSM
mnemonics_unique_df[mnemonics_unique_df["mnemonic"] == "DTCO"]
mnemonics_unique_df[mnemonics_unique_df["mnemonic"] == "DTSM"]


#%% group similar mnemonics


def text_similar_score(a: list, b: list) -> float:
    """return the similarity between two sentences a, b"""
    from difflib import SequenceMatcher

    assert type(a) is str
    assert type(b) is str

    a = "".join(a).lower().replace(" ", "")
    b = "".join(b).lower().replace(" ", "")
    return SequenceMatcher(None, a, b).ratio()


mnemonics_groups = []
for _, a, b in mnemonics_unique_df.itertuples():
    mnemonics_groups.append(a + "*" + b)

print("total number of ungrouped mnemonics:", len(mnemonics_groups))

mnemonics_groups_2 = []
mnemonics_groups_2_ = []

for m in mnemonics_groups:

    if m not in mnemonics_groups_2_:
        m_group = [m]
        mnemonics_groups_2_.append(m)

        for m_ in mnemonics_groups:
            if (
                (m_ != m)
                and (text_similar_score(m, m_) > 0.5)
                and (m_ not in mnemonics_groups_2_)
            ):
                m_group.append(m_)
                mnemonics_groups_2_.append(m_)
        mnemonics_groups_2.append(m_group)

print("total looped mnemonics:", len(mnemonics_groups_2_))
print("total unique mnemonics:", len(np.unique(mnemonics_groups_2_)))
print("total groups of mnemonics:", len(mnemonics_groups_2))
print("total unique group of mnemonics:", len(np.unique(mnemonics_groups_2)))

mnemonics_groups_3 = []
for m_group in mnemonics_groups_2:
    for m_group_ in m_group:
        mnemonics_groups_3.append(m_group_.split("*"))
    mnemonics_groups_3.append(["-" * 20, "------" * 20])

mnemonics_groups_3 = pd.DataFrame(
    mnemonics_groups_3, columns=["mnemonic", "description"]
)
mnemonics_groups_3.to_csv("data/grouped_mnemonics.csv")
