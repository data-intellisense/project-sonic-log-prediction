import pandas as pd
import numpy as np
import seaborn 
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go 

pio.renderers.default = 'browser'

# import cordinates
cords = pd.read_csv('data/cords.csv', index_col=0)

print(cords.sample(5))

fig = go.Figure()
fig.add_traces(go.Scatter(x=cords['Lon'], y=cords['Lat'], mode='markers', marker=dict(size=cords['STOP']/500),
                    hoverinfo='text', hovertext = cords['Well']))
fig.update_layout(xaxis = dict(title='Longitude'),
                    yaxis = dict(title='Latitude'),
                    title = dict(text='Size: Stop Depth'),
                    font=dict(size=18))



