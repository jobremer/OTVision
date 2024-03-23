
data = OTCdetections[OTCdetections['class'] == 'pedestrian']
data = data[data['frame'] >= 10500]
data = data[data['frame'] <= 11000]
data = data[data['track-id'] == 966]

# Hintergrundbild
background = mpimg.imread(dirpath + "Screenshot.png")

# Den DataFrame nach 'track-id' gruppieren
grouped_df = data.groupby('track-id')

# Iteration durch die Gruppen und Plotten der Trajektorien
for group_name, group_df in grouped_df:
    x = group_df['x']
    y = group_df['y']
    # plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=3, color='yellow')
    plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=2, color='yellow')

# Plot-Einstellungen
plt.rcParams["figure.figsize"] = [10,10]
plt.rcParams["figure.autolayout"] = True
plt.xlabel('x-Koordinate', size=20) 
plt.ylabel('y-Koordinate', size=20)
# plt.legend(fontsize=10)

# plt.xlim(0, 1296)
# plt.ylim(0, 972)
plt.xlim(575, 675)
plt.ylim(280, 350)
plt.gca().invert_yaxis()
plt.imshow(background)
plt.savefig("Z:/Masterthesis/Images/08a_Trajektorienfluktuationen.png")

plt.show()