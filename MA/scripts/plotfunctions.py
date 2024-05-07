import numpy as np

def ottrk_to_txt(filepath, key1 = 'data', key2 = 'detections', export = False):
    import pandas as pd
    import bz2
    import ujson
    print('ottrk_to_txt...')
    
    # Open ottrk-file
    with bz2.open(filepath, "rt", encoding="UTF-8") as file:
        dictfile = ujson.load(file)
    
    # Write in DataFrame
    detections = pd.DataFrame.from_dict(dictfile['data']['detections'])
    detections = detections[['x', 'y', 'w', 'h', 'frame', 'track-id', 'class']] #bb_left = x, bb_top = y
    
    # Mittelpunkte der BBOxes
    detections['x0'] = detections['x'] + detections['w'] / 2
    detections['y0'] = detections['y'] + detections['h'] / 2
    
    metadata = pd.DataFrame.from_dict(dictfile['metadata'])

    detections['sec'] = (detections['frame'] - detections['frame'].min()) / 20 #20fps
    detections = detections.sort_values(by=['track-id', 'frame'])
    
    # Export
    if export == True:
        detections.to_csv(filepath + 'detections' + '.csv', encoding='utf-8', header=True, index=False)

    return detections, metadata


def tracklength(detections):
    import pandas as pd
    import numpy as np
    print('tracklength...') 
    
    detections = detections.sort_values(by=['track-id', 'frame'])
    detections['dx'] = 0
    detections['dy'] = 0

    # Richtungsvektor
    detections['dx'] = detections['x0'].diff()
    detections['dy'] = detections['y0'].diff()
    detections['d_l'] = np.sqrt(np.power(detections['dx'],2) + np.power(detections['dy'],2))

    # Ersten Wert von d_l für jede track.-id = 0 setzen
    def replace_first_value(group):
        if group.iloc[0] != 0:
            group.iloc[0] = 0
        return group

    detections['dx'] = detections.groupby('track-id')['dx'].apply(replace_first_value).reset_index(level=0, drop=True)
    detections['dy'] = detections.groupby('track-id')['dy'].apply(replace_first_value).reset_index(level=0, drop=True)
    detections['d_l'] = detections.groupby('track-id')['d_l'].apply(replace_first_value).reset_index(level=0, drop=True)

    tracklength = pd.DataFrame(detections.groupby('track-id')['d_l'].sum().sort_values(ascending = False))
    tracklength['track-length'] = tracklength['d_l']
    del tracklength['d_l']

    detections = pd.merge(detections, tracklength, on='track-id')
    
    begin = detections.groupby(['track-id'])['frame'].min().reset_index()
    begin['Position'] = 'begin'
    end = detections.groupby(['track-id'])['frame'].max().reset_index()
    end['Position'] = 'end'
    detections = pd.merge(detections, pd.concat([begin, end]), how='left', on = ['track-id', 'frame'])
    del begin, end
    
    return detections
    
def plot_histogram(detections, xlim = 0, binwidth = 100, fontsize = 15, xText = "", titletext = "", savename = ""):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    import pandas as pd
    
    plt.hist(detections.groupby(['track-id'])['track-length'].first(),
             bins=range(min(detections['track-length'].astype('int')), max(detections['track-length'].astype('int')) + binwidth, binwidth),
             edgecolor='black')
    if xlim != 0:
        plt.xlim(0, xlim)
    plt.rcParams["figure.figsize"] = [20,10]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel(xText, size=fontsize, weight = 'bold') 
    plt.ylabel('Anzahl', size=fontsize, weight = 'bold')
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    if titletext !="":
        plt.title(titletext, fontsize=fontsize+4, weight='bold')
    
    if savename != "":
        plt.savefig("Z:/Masterthesis/Images/" + savename + ".png")
    

def plot_trajectory(detections, filepath, box, fontsize = 15, titletext = "", show_background = False, savename = "", marker = False):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    
    print(box)

    # Den DataFrame nach 'track-id' gruppieren
    grouped_df = detections.groupby('track-id')

    # Iteration durch die Gruppen und Plotten der Trajektorien
    for group_name, group_df in grouped_df:
        x = group_df['x0']
        y = group_df['y0']
        
        if marker == True:
            plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=1, marker='.', markersize=0.7)
        else:
            plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=1)
        # plt.plot(x, y, 'o', markersize=0.5)
        # plt.plot(x, y, label=f'Track ID: {group_name}', linewidth=1)

    # Plot-Einstellungen
    plt.rcParams["figure.figsize"] = [10,10]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel('x-Koordinate', size=fontsize, weight = 'bold') 
    plt.ylabel('y-Koordinate', size=fontsize, weight = 'bold')
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    # plt.legend(fontsize=8)

    # plt.xlim(0, 1296)
    # plt.ylim(0, 972)
    plt.xlim(0, box[0])
    plt.ylim(0, box[1])
    plt.gca().invert_yaxis()
    
    # Hintergrundbild
    if show_background == True:
        background = mpimg.imread(filepath + "/Screenshot.png")
        plt.imshow(background)
    
    if titletext !="":
        plt.title(titletext, fontsize=fontsize+4, weight='bold')

    if savename != "":
        plt.savefig("Z:/Masterthesis/Images/" + savename + ".png")



def plot_BBoxes(data, filepath,
                box: list, 
                track_id: int = np.NaN, 
                framerate: int = 20, 
                fontsize: int = 15, 
                include_trajectories: bool = False,
                custom_picture_section: list = [],
                include_legend: bool = False,
                titletext: str = "", 
                savename: str = ""):
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    from matplotlib.patches import Rectangle
    from PIL import Image
    import numpy as np
    
    if track_id == np.NaN:
        detections = data
    else:
        detections = data[data['track-id'] == track_id]
    
    detections = data
    
    detections_reduced = detections.iloc[::framerate].reset_index(drop=True)

    # Hintergrundbild
    background = mpimg.imread(filepath + "/Screenshot.png")

    # Farben
    farbpalette = plt.colormaps['tab10']

    for index, row in detections_reduced.iterrows():
        x = row['x']
        y = row['y']
        h = row['h']
        w = row['w']
        
        # Masse der Boundding Box
        width = box[0]
        height = box[1]
        
        farbe = farbpalette(index % farbpalette.N)
        
        # Rechteck zeichnen
        plt.gca().add_patch(Rectangle((x, y), w, h, edgecolor=farbe, facecolor='none', linewidth=2))
        plt.plot(x, y, linewidth=2, color='yellow')
        
    if include_trajectories == True:
        plot_trajectory(detections=detections, filepath=filepath, box=box)
   

    # Plot-Einstellungen
    plt.rcParams["figure.figsize"] = [10,5]
    plt.rcParams["figure.autolayout"] = True
    plt.xlabel('x-Koordinate', size=fontsize, weight = 'bold') 
    plt.ylabel('y-Koordinate', size=fontsize, weight = 'bold')
    plt.xticks(fontsize=fontsize-2)
    plt.yticks(fontsize=fontsize-2)
    
    if include_legend == True:
        plt.legend(fontsize=10)

    # plt.xlim(0, 1296)
    # plt.ylim(0, 972)
    
    if custom_picture_section != []:
        plt.xlim(custom_picture_section[0], custom_picture_section[1])
        plt.ylim(custom_picture_section[2], custom_picture_section[3])
    else:
        plt.xlim(0, width)
        plt.ylim(0, height)
        
    plt.gca().invert_yaxis()
    plt.imshow(background)
    
    if titletext !="":
        plt.title(titletext, fontsize=fontsize+4, weight='bold')

    if savename != "":
        plt.savefig("Z:/Masterthesis/Images/" + savename + ".png")
        
    return detections_reduced