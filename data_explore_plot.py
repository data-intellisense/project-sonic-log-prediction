#%% import 
import pickle
import numpy as np
import pandas as pd
import lasio
import plotly.express as px
import plotly.io as pio

from util import get_mnemonic, get_alias,read_las, process_las
from plot import plot_logs_columns, plot_crossplot
from sklearn.neighbors import LocalOutlierFactor

pio.renderers.default = 'browser'

from load_pickle import alias_dict, las_data_DTSM_QC
# las_data_DTSM['000-0052442d0162_TGS']

#%% plot all las

# for key in las_data.keys():
#     plot_logs_columns(las_data[key],
#                       well_name=key,
#                       plot_show=False,
#                       plot_save_file_name=key,
#                       plot_save_path='plots/plots_las',
#                       plot_save_format=['png', 'html'])


# #%% plot all las with DTSM

# for key in las_data_DTSM.keys():
#     plot_logs_columns(las_data_DTSM[key],
#                       well_name=key,
#                       plot_show=False,
#                       plot_save_file_name=f'{key}_DTSM',
#                       plot_save_path='plots/plots_las',
#                       plot_save_format=['png', 'html'])



#%% plot all las with new QC'd DTSM

# for key in las_data_DTSM_QC.keys():
#     plot_logs_columns(las_data_DTSM_QC[key],
#                       well_name=key,
#                       plot_show=False,
#                       plot_save_file_name=f'{key}_DTSM',
#                       plot_save_path='plots/plots_las_DTSM_QC',
#                       plot_save_format=['png'])



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

target_mnemonics = ['DTCO', 'RHOB', 'NPHI',  'GR', 'CALI', 'RT', 'PEFZ']
target_mnemonics = target_mnemonics + ['DTSM']

for key in las_data_DTSM_QC.keys():
    
    df = las_data_DTSM_QC[key]
    print('processing:\t', key)

    for m in target_mnemonics:
        
        alias = [ i for i in get_alias(m) if i in df.columns]
        alias_count = len(alias)
        if m not in df_target_mnemonics_count.keys():
            df_target_mnemonics_count[m] = [[key[:3], alias_count]]
        else:
            df_target_mnemonics_count[m].append([key[:3], alias_count])
        
        if alias_count >=2:
            if key not in df_target_mnemonics_count2.keys():
                df_target_mnemonics_count2[key] = [alias] 
            else:
                df_target_mnemonics_count2[key] += [alias] 

        for col in df.columns:        
            if get_mnemonic(col, alias_dict=alias_dict) == m:
                arr = df[col].values.reshape(-1,1)
                arr_avg = np.mean(arr)
                if m not in df_target_mnemonics.keys():
                    df_target_mnemonics[m] = arr
                    df_target_mnemonics_avg[m] = arr_avg                   
                    
                else:
                    df_target_mnemonics[m] = np.r_[df_target_mnemonics[m], arr]
                    df_target_mnemonics_avg[m] = np.r_[df_target_mnemonics_avg[m], arr_avg]
                    
                
                # check any caliper average smaller than 7
                if (arr_avg<7 or arr_avg>12) and m == 'CALI':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any DTCO average more than 100
                if arr_avg>=100 and m == 'DTCO':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any DTCO average more than 100
                if arr_avg>=180 and m == 'DTSM':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any NPHI higher than .25                
                if arr_avg>=0.25 and m == 'NPHI':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any RT higher than 2000
                if arr_avg>=2000 and m == 'RT':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                       

#%% plot histogram for each target mnemonic

for key in df_target_mnemonics.keys():
    df_target_mnemonics[key] = pd.DataFrame(df_target_mnemonics[key], columns=[key])
    fig=px.histogram(df_target_mnemonics[key], x=key, title=key)
    fig.update_layout(        
        showlegend=True,        
        title=dict(text=key, font=dict(size=20)),
        font=dict(size=18),
        template='plotly',        
        width=3000,
        height=1200,
    )
    fig.write_image(f'plots/plots_histogram/histogram_{key}.png')
    print(df_target_mnemonics[key].shape)

    df_target_mnemonics_avg[key] = pd.DataFrame(df_target_mnemonics_avg[key], columns=[key])
    fig2=px.histogram(df_target_mnemonics_avg[key], x=key, title=f'{key}-Average')
    fig2.update_layout(        
        showlegend=True,        
        title=dict(text=f'{key}-Average', font=dict(size=20)),
        font=dict(size=18),
        template='plotly',        
        width=3000,
        height=1200,
    )
    fig2.write_image(f'plots/plots_histogram/histogram_{key}_avg.png')


    df_target_mnemonics_count[key] = pd.DataFrame(np.array(df_target_mnemonics_count[key]), columns=['las#', key])
    fig3=px.histogram(df_target_mnemonics_count[key], x=key, title=f'{key}-Count')
    fig3.update_layout(        
        showlegend=True,        
        title=dict(text=f'{key}-Count', font=dict(size=20)),
        font=dict(size=18),
        template='plotly',        
        width=3000,
        height=1200,
    )
    fig3.write_image(f'plots/plots_histogram/histogram_{key}_count.png')

df_target_mnemonics_count
len(df_target_mnemonics_count2)

#%%
df_target_mnemonics = dict()
df_target_mnemonics_avg = dict()
df_target_mnemonics_count = dict()
df_target_mnemonics_count2 = dict()
las2qc = dict()

target_mnemonics = ['DTCO', 'NPHI', 'RHOB', 'GR', 'CALI', 'RT', 'PEFZ']
target_mnemonics = target_mnemonics + ['DTSM']

for key in las_data_DTSM_QC.keys():
    
    df = las_data_DTSM_QC[key]
    print('processing:\t', key)

    for m in target_mnemonics:
        
        alias = [ i for i in get_alias(m) if i in df.columns]
        alias_count = len(alias)
        if m not in df_target_mnemonics_count.keys():
            df_target_mnemonics_count[m] = [[key[:3], alias_count]]
        else:
            df_target_mnemonics_count[m].append([key[:3], alias_count])
        
        if alias_count >=2:
            if key not in df_target_mnemonics_count2.keys():
                df_target_mnemonics_count2[key] = [alias] 
            else:
                df_target_mnemonics_count2[key] += [alias] 

        for col in df.columns:        
            if get_mnemonic(col, alias_dict=alias_dict) == m:
                arr = df[col].values.reshape(-1,1)
                arr_avg = np.mean(arr)
                if m not in df_target_mnemonics.keys():
                    df_target_mnemonics[m] = arr
                    df_target_mnemonics_avg[m] = arr_avg                   
                    
                else:
                    df_target_mnemonics[m] = np.r_[df_target_mnemonics[m], arr]
                    df_target_mnemonics_avg[m] = np.r_[df_target_mnemonics_avg[m], arr_avg]
                    
                
                # check any caliper average smaller than 7
                if (arr_avg<7 or arr_avg>12) and m == 'CALI':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any DTCO average more than 100
                if arr_avg>=100 and m == 'DTCO':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                # check any DTCO average more than 100
                if arr_avg>=180 and m == 'DTSM':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any NPHI higher than .25                
                if arr_avg>=0.25 and m == 'NPHI':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)
                
                # check any RT higher than 2000
                if arr_avg>=2000 and m == 'RT':
                    if m not in las2qc.keys():
                        las2qc[m] = [key]
                    else:
                        las2qc[m].append(key)

                       
#%% plot crossplot

mnemonics_x = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]
mnemonics_y = ["DTCO", "RHOB", "NPHI", "GR", "RT", "CALI", "PEFZ"]


target_mnemonics_ = [[x, y] for x in mnemonics_x for y in mnemonics_y if x!=y]

for target_mnemonics in target_mnemonics_:
    las_dict = dict()
    for key in las_data_DTSM_QC.keys():
        
        df = las_data_DTSM_QC[key]
        print('processing:\t', key)

        df = process_las().despike(df, window_size=5)

        df = process_las().get_df_by_mnemonics(
            df=df, target_mnemonics=target_mnemonics, strict_input_output=True, alias_dict=alias_dict
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
        plot_save_format=["png", 'html']
    )