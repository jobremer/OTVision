import sys
import os
import bz2
import csv
import pandas as pd
import ast
# import json
import ujson
import numpy

def events_to_df(filepath, key = 'event_list'):
    # Open otevents-file
    with bz2.open(filepath, "rt", encoding="UTF-8") as file:
        dictfile = ujson.load(file)
    
    # Convert to DataFrame
    EVENTS = pd.DataFrame.from_dict(dictfile[key])

    return EVENTS

def ottrk_to_mot(filename, key1 = 'data', key2 = 'detections', directory = "convert_tracks/OTCamera/"):
    print(filename + ' wird konvertiert...')
    fileending = '.ottrk'
    filepath = os.path.join(directory, filename) + fileending

    # Open ottrk-file
    with bz2.open(filepath, "rt", encoding="UTF-8") as file:
        dictfile = ujson.load(file)
    
    # Write in DataFrame
    detections = pd.DataFrame.from_dict(dictfile[key1][key2])
    detections = detections[['x', 'y', 'w', 'h', 'frame', 'track-id']] #bb_left = x, bb_top = y


    # Transform to MOTChallenge-format (x,y,z are ignored fpr 2D-challenges)
    detections['conf'] = 1
    detections['3D_x'] = -1
    detections['3D_y'] = -1
    detections['3D_z'] = -1
    detections = detections[['frame', 'track-id', 'x', 'y', 'h', 'w', 'conf', '3D_x', '3D_y', '3D_z']]

    # # Export
    print('Export nach ' + directory + '/det/')
    detections.to_csv(directory + '/det/' + filename + '.txt', encoding='utf-8', header=False, index=False)
 
    return detections

# OTCdetections, MOTdetections = ottrk_to_mot(filename = "OTCamera13_FR20_2023-10-22_17-00-00_Sued")

# def mot_gt_to_otdet(filename):

