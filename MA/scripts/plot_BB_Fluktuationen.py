from matplotlib.patches import Rectangle
from PIL import Image

data = OTCdetections[OTCdetections['class'] == 'pedestrian']
# data = data[data['frame'] >= 10500]
# data = data[data['frame'] <= 11000]
data = data[data['track-id'] == 966]
data = data[data['x'] >= 500]
data = data[data['x'] <= 700]

data['x0'] = data['x'] - data['w'] / 2
data['y0'] = data['y'] - data['h'] / 2
data2 = data.iloc[::20].reset_index(drop=True)

# Hintergrundbild
background = mpimg.imread(dirpath + "Screenshot.png")

# Den DataFrame nach 'track-id' gruppieren
grouped_df = data.groupby('frame')

farbpalette = plt.colormaps['tab10']

for index, row in data2.iterrows():
    x = row['x']
    y = row['y']
    h = row['h']
    w = row['w']
    
    # Berechnung der Koordinaten der Ecken des Rechtecks
    x0 = x - w/2
    y0 = y - 2*h/9
    
    farbe = farbpalette(index % farbpalette.N)
    
    # Rechteck zeichnen
    plt.gca().add_patch(Rectangle((x0, y0), w, h, edgecolor=farbe, facecolor='none', linewidth=3))
    plt.plot(x, y, linewidth=2, color='yellow')
# Den DataFrame nach 'track-id' gruppieren
grouped_df = data.groupby('track-id')

# Iteration durch die Gruppen und Plotten der Trajektorien
for group_name, group_df in grouped_df:
    x = group_df['x']
    y = group_df['y']
    # plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=3, color='yellow')
    plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=2, color='yellow')

# Plot-Einstellungen
plt.rcParams["figure.figsize"] = [10,5]
plt.rcParams["figure.autolayout"] = True
plt.xlabel('x-Koordinate', size=12) 
plt.ylabel('y-Koordinate', size=12)
# plt.legend(fontsize=10)

# plt.xlim(0, 1296)
# plt.ylim(0, 972)
plt.xlim(575, 675)
plt.ylim(280, 350)
plt.gca().invert_yaxis()
plt.imshow(background)
plt.savefig("Z:/Masterthesis/Images/08_BoundingBox_Fluktuationen.png")

plt.show()

# del group_df, group_name, grouped_df
