import sys
import os
import bz2
import csv
import pandas as pd
import ast
# import json
import ujson
import numpy

def ottrk_to_mot(filename, key1 = 'data', key2 = 'detections'):
    print(filename + ' wird konvertiert...')
    dirpath = "convert_tracks/OTCamera/"
    fileending = '.ottrk'
    filepath = os.path.join(dirpath, filename) + fileending

    # Open ottrk-file
    with bz2.open(filepath, "rt", encoding="UTF-8") as file:
        dictfile = ujson.load(file)
    
    # Write in DataFrame
    OTCdetections = pd.DataFrame.from_dict(dictfile[key1][key2])
    MOTdetections = detections[['x', 'y', 'w', 'h', 'frame', 'track-id']] #bb_left = x, bb_top = y

    # Transform to MOTChallenge-format (x,y,z are ignored fpr 2D-challenges)
    MOTdetections['conf'] = 1
    MOTdetections['3D_x'] = -1
    MOTdetections['3D_y'] = -1
    MOTdetections['3D_z'] = -1
    MOTdetections = MOTdetections[['frame', 'track-id', 'x', 'y', 'h', 'w', 'conf', '3D_x', '3D_y', '3D_z']]

    # Export
    MOTdetections.to_csv(dirpath + filename + '.txt', encoding='utf-8', header=False, index=False)
 
    return OTCdetections, MOTdetections

# OTCdetections, MOTdetections = ottrk_to_mot(filename = "OTCamera13_FR20_2023-10-22_17-00-00_Sued")

# def mot_gt_to_otdet(filename):