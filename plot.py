import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from util import alias_dict, get_mnemonic, get_alias

from sklearn.metrics import mean_squared_error
import random 
import pickle 

# with open('data/alias_dict.pickle', 'rb') as f:
#     alias_dict = pickle.load(f)

pio.renderers.default = "browser"

#%% simple plot of all curves in one column, good for quick plotting

def plot_logs(
    df,
    well_name="well",
    DTSM_only=True,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
):

    '''
    simple plot of all curves in one column, good for quick plotting
    '''

    df = df.copy()

    # drop part of logs where no DTSM data
    if DTSM_only:
        df.dropna(subset=["DTSM"], inplace=True)

    fig = go.Figure()

    for col in df.columns:

        # do not plots in the below list:
        if col not in ["DEPT", "TEND", "TENR"]:
            fig.add_trace(go.Scatter(x=df[col], y=df.index, name=col))

    fig.update_layout(
        showlegend=True,
        title=dict(text=well_name),
        yaxis=dict(autorange="reversed", title="Depth, ft"),
        font=dict(size=18),
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


#%% complex plot that plots curves in multiple columns

def plot_logs_columns(
    df,
    DTSM_pred=None,
    well_name="",
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"]
):
    '''
    complex plot that plots curves in multiple columns, good for detailed analysis of curves
    '''

    df = df.copy()

    # determine how many columns for grouped curves
    columns = df.columns.map(alias_dict)
    tot_cols = [['DTCO', 'DTSM'],                   #  row=1, col=1
                ['RHOB'],                           #  row=1, col=2
                ['DPHI'],                           #  row=1, col=3
                ['NPHI'],                           #  row=1, col=4
                ['GR'],                             #  row=1, col=5
                ['AT', 'RT'],                       #  row=1, col=6
                ['CALI'],                           #  row=1, col=7
                ['PEFZ']]                           #  row=1, col=8
    
    num_of_cols = 1    
    tot_cols_new = [] # update the tot_cols if some curves are missing
    tot_cols_old = []
    
    for cols in tot_cols:
        if any([(i in columns) for i in cols]):
            tot_cols_new.append(cols)            
            num_of_cols +=1    

        # get the old columns as subplot titles
        temp = []
        for i in df.columns:
                        
            if get_mnemonic(i) in cols: 
                temp.append(i)
        if len(temp)>0:
            tot_cols_old.append(temp)

    # make subplots (flexible with input)
    fig = make_subplots(rows=1, cols=num_of_cols, subplot_titles=[','.join(j) for j in tot_cols_old], shared_yaxes=True)

    for col_old in df.columns:

        # find the mnemonic for alias
        col_new = get_mnemonic(col_old, alias_dict=alias_dict)
        try:
            # find the index for which column to plot the curve            
            col_id = [i+1 for i, v in enumerate(tot_cols_new) if col_new in v][0]
        except:
            col_id = num_of_cols
               
        # print(f'col_old: {col_old}, col_new: {col_new}, col_id: {col_id}')
        # if 'TENS' not in col_new:
        fig.add_trace(go.Scatter(x=df[col_old], y=df.index, name=col_old), row=1, col=col_id)
                       
    # add predicted DTSM if not None
    if DTSM_pred is not None:
        fig.add_trace(go.Scatter(x=DTSM_pred['DTSM_Pred'], y=DTSM_pred['Depth'], line_color='rgba(255, 0, 0, .7)', name='DTSM_Pred'), row=1, col=1)             

    fig.update_layout(        
        showlegend=True,        
        title=dict(text=well_name, font=dict(size=12)),
        yaxis=dict(autorange="reversed", title="Depth, ft"),
        font=dict(size=18),
        legend=dict(orientation='h',
                    y=1.07, yanchor='middle',
                    x=0.5, xanchor='center',
                    font=dict(size=12)),
        template='plotly',        
        width=3000,
        height=1200,
    )

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if plot_save_path is not None:

                if not os.path.exists(plot_save_path):
                    os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig

#%% plot predicted and actual in a crossplot
def plot_crossplot(y_actual, y_predict, text=None,
    axis_range = 350,
    plot_show=True,
    plot_return=False,
    plot_save_file_name=None,
    plot_save_path=None,
    plot_save_format=None,  # availabe format: ["png", "html"])
):
    
    assert len(y_actual)==len(y_predict)
    rmse_test =  mean_squared_error(y_actual, y_predict)**0.5    
    
    y_pred_act = pd.DataFrame(np.c_[y_actual.reshape(-1,1), y_predict.reshape(-1,1)], columns=['Actual', 'Predict'])
    abline = pd.DataFrame(np.c_[np.arange(axis_range).reshape(-1,1), np.arange(axis_range).reshape(-1,1)], columns=['Actual', 'Predict'])

    if text is not None:
        title_text = f'{text}, rmse_test:{rmse_test:.2f}'
    else:
        title_text = f'rmse_test:{rmse_test:.2f}'

    fig = px.scatter(y_pred_act, x='Actual', y='Predict')
    fig.add_traces(px.line(abline, x='Actual', y='Predict').data[0])
    fig.update_layout(title=dict(text=title_text),
                        width=1200,
                        height=1200,
                        xaxis_range=[0,axis_range],
                        yaxis_range=[0,axis_range])
    
    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig


def plot_cords(cords=None,
                plot_show=True,
                plot_return=False,
                plot_save_file_name=None,
                plot_save_path=None,
                plot_save_format=None,  # availabe format: ["png", "html"]
                ):
        
    fig = go.Figure()
    fig.add_traces(go.Scatter(x=cords['Lon'], y=cords['Lat'], mode='markers'),
                        hoverinfo='text', hovertext = cords['Well'])
    fig.update_layout(xaxis = dict(title='Longitude'),
                        yaxis = dict(title='Latitude'),
                        # title = dict(text='Size: Stop Depth'),
                        font=dict(size=18))

    # show and save plot
    if plot_show:
        fig.show()

    # save the figure if plot_save_format is provided
    if plot_save_format is not None:

        if plot_save_file_name is None:
            plot_save_file_name = f"plot-{str(np.random.random())[2:]}"

        if plot_save_path is not None:
            if not os.path.exists(plot_save_path):
                os.mkdir(plot_save_path)

            plot_save_file_name = f"{plot_save_path}/{plot_save_file_name}"
            print(f"\nPlots are saved at path: {plot_save_path}!")
        else:
            print(f"\nPlots are saved at the same path as current script!")

        for fmt in plot_save_format:

            plot_file_name_ = f"{plot_save_file_name}.{fmt}"

            if fmt in ["png"]:
                fig.write_image(plot_file_name_)
            if fmt in ["html"]:
                fig.write_html(plot_file_name_)

    if plot_return:
        return fig