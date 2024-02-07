# import plotly.express as px
import numpy as np
import convert_motchallenge as ottrk_to_mot
# import datashader as ds
# import datashader.transfer_functions as tf
import matplotlib

OTCdetections, MOTdetections = ottrk_to_mot(filename = "OTCamera13_FR20_2023-10-22_17-00-00_Sued")

# Frames filtern
data = OTCdetections
# data = detections[detections['frame'] >=3380]
# data = data[data['frame'] <=5000]
# data = data[data['class'] == 'delivery_van']
# data['track-id'].astype(str)

data['sec'] = (data['frame'] - data['frame'].min()) / 20 #20fps

# print(data['track-id'].value_counts())
# data = data[data['track-id'] == 14476]



plotdata = data
plotdata['x'] = plotdata['x'] * -1
plotdata['y'] = plotdata['y']
# Default plot ranges:
x_range = (plotdata['x'].min(), plotdata['x'].max())
y_range = (plotdata['y'].min(), plotdata['y'].max())
print(x_range)
print(y_range)
# x_range = (0, 1000)
# y_range = (-1000, 0)
plotdata = plotdata.sort_values(by=['track-id', 'frame'])

res = plotdata.set_index('track-id').groupby('track-id').apply(
    lambda x: x.reset_index(drop=True).reindex(np.arange(len(x) + 1)))

def create_image(w=500, h=500):
    cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
    agg = cvs.line(plotdata, 'x', 'y', agg=ds.any())
    return tf.shade(agg, cmap = 'white')

create_image()