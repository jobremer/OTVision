"""
OTVision module to track road users in frames detected by OTVision
"""
# based on IOU Tracker written by Erik Bochinski originally licensed under the
# MIT License, see
# https://github.com/bochinski/iou-tracker/blob/master/LICENSE.

# Copyright (C) 2022 OpenTrafficCam Contributors
# <https://github.com/OpenTrafficCam
# <team@opentrafficcam.org>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from tqdm import tqdm

from OTVision.config import CONFIG
from OTVision.dataformat import (
    AGE,
    BBOXES,
    CENTER,
    CLASS,
    CONFIDENCE,
    DETECTIONS,
    FINISHED,
    FIRST,
    FRAMES,
    MAX_CLASS,
    MAX_CONF,
    START_FRAME,
    TRACK_ID,
    H,
    W,
    X,
    Y,
    X_VECTOR,
    Y_VECTOR,
    VECTOR_AMOUNT,
    DIRECTION_VECTOR,
    VECTOR_AMOUNT_ROLLING_MEAN,
    X_VECTOR_ROLLING_MEAN,
    Y_VECTOR_ROLLING_MEAN,
    VECTOR_ROLLING_MEAN,
    CENTER_EXTRAPOLATED,
    BBOXES_EXTRAPOLATED,
    DIMENSIONS,
    BBOXES_ROLLING_MEAN,
    MODE,
)

from .iou_util import iou

# New packages
import numpy as np
import scipy as sp
from scipy.ndimage import uniform_filter1d


def make_bbox(obj: dict) -> tuple[float, float, float, float]:
    """Calculates xyxy coordinates from dict of xywh.

    Args:
        obj (dict): dict of pixel values for xcenter, ycenter, width and height

    Returns:
        tuple[float, float, float, float]: xmin, ymin, xmay, ymax
    """
    
    return (
        obj[X] - obj[W] / 2,
        obj[Y] - obj[H] / 2,
        obj[X] + obj[W] / 2,
        obj[Y] + obj[H] / 2,
    )
    
def make_extrapolated_bbox(obj: list) -> list[float, float, float, float]:
    """Calculates xyxy coordinates from dict of xywh.

    Args:
        obj (dict): dict of pixel values for xcenter, ycenter, width and height

    Returns:
        tuple[float, float, float, float]: xmin, ymin, xmay, ymax
    """
    
    liste = [obj[0] - obj[2] / 2,
             obj[1] - obj[3] / 2,
             obj[0] + obj[2] / 2,
             obj[1] + obj[3] / 2]
        
    
    return (liste)


def center(obj: dict) -> tuple[float, float]:
    """Retrieves center coordinates from dict.

    Args:
        obj (dict): _description_

    Returns:
        tuple[float, float]: _description_
    """
    return obj[X], obj[Y]


def get_direction_vector(track, best_match):
    """
    Calculate the x- & y- vectors and the vector amount for the current track and its best match

    Args:
        track (dict)
        best_match (dict)

    Returns:
        list: [float, float, float]
    """
    vector = [track[CENTER][-1], [best_match['x'], best_match['y']]]
    X_VECTOR = vector[-1][0] - vector[0][0]
    Y_VECTOR = vector[-1][1] - vector[0][1]
    AMOUNT_VECTOR = np.sqrt(np.square(X_VECTOR) + np.square(Y_VECTOR))
    
    return X_VECTOR, Y_VECTOR, AMOUNT_VECTOR

def get_rolling_mean(first, second,
                     t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"]):
    x_cumsum = sum(first)
    y_cumsum = sum(second)
    
    return [x_cumsum / float(t_min), y_cumsum / float(t_min)]

def get_rolling_mean_BBox(first, second,
                     t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"]):
    x_cumsum = sum(first[-1:])
    y_cumsum = sum(second[-1:])
    
    return [x_cumsum / float(t_min), y_cumsum / float(t_min)]

# def get_rolling_mean(first, second, t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"]):
#     nth_last_index = max(len(first) - 2, 0)  # Index des fünftletzten Elements
#     t_min = min(t_min, len(first) - nth_last_index)  # Aktualisieren von t_min, um sicherzustellen, dass es nicht größer ist als die Anzahl der Elemente ab dem fünftletzten Element
#     x_cumsum = sum(first[nth_last_index:])  # Die Summe beginnt beim fünftletzten Element
#     y_cumsum = sum(second[nth_last_index:])  # Die Summe beginnt beim fünftletzten Element
    
#     return [x_cumsum / float(t_min), y_cumsum / float(t_min)]

def get_rolling_mean_angle(vectors): # Written with help of ChatGPT3.5
    angles = []
    if len(vectors) > 1:
        for i in range(len(vectors) - 1):
            vec1 = vectors[i]
            vec2 = vectors[i + 1]

            radians = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
            degrees = np.degrees(radians)
            # Konvertierung negativer Winkel in positive Winkel
            # if degrees > 180:
            #     degrees = degrees - 360
            # elif degrees < -180:
            #     degrees = degrees + 360
            
            angles.append(degrees)
    
    else:
        angles.append(0)
    
    # Berechne den Mittelwert der letzten fünf Winkel
    last_five_angles = angles[-5:]
    mean_angle = np.median(angles[-5:])
    return mean_angle

def rotate(vector, angle_degree): # Rotate vector
    (x,y) = vector
    angle_radian = angle_degree*np.pi/180
    newx = x*np.cos(angle_radian) - y*np.sin(angle_radian)
    newy = x*np.sin(angle_radian) + y*np.cos(angle_radian)
    return [newx, newy]
 

   

# def get_rolling_mean3(first, second, t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"]):
#     nth_last_index = max(len(first) - 5, 0)  # Index des fünftletzten Elements
#     t_min = min(t_min, len(first) - nth_last_index)  # Aktualisieren von t_min, um sicherzustellen, dass es nicht größer ist als die Anzahl der Elemente ab dem fünftletzten Element
#     x_cumsum = sum(first[nth_last_index:])  # Die Summe beginnt beim fünftletzten Element
#     y_cumsum = sum(second[nth_last_index:])  # Die Summe beginnt beim fünftletzten Element
    
#     return [x_cumsum / float(t_min), y_cumsum / float(t_min)]


def track_iou(
    detections: list,  # TODO: Type hint nested list during refactoring
    sigma_l: float = CONFIG["TRACK"]["IOU"]["SIGMA_L"],
    sigma_h: float = CONFIG["TRACK"]["IOU"]["SIGMA_H"],
    sigma_iou: float = CONFIG["TRACK"]["IOU"]["SIGMA_IOU"],
    t_min: int = CONFIG["TRACK"]["IOU"]["T_MIN"],
    t_miss_max: int = CONFIG["TRACK"]["IOU"]["T_MISS_MAX"],
    t_extrapolate: int = CONFIG["TRACK"]["IOU"]["T_EXTRAPOLATE"],
) -> dict:  # sourcery skip: low-code-quality
    """
    Simple IOU based tracker.
    See "High-Speed Tracking-by-Detection Without Using Image Information
    by E. Bochinski, V. Eiselein, T. Sikora" for
    more information.

    Args:
        detections (list): list of detections per frame, usually generated
        by util.load_mot
        sigma_l (float): low detection threshold.
        sigma_h (float): high detection threshold.
        sigma_iou (float): IOU threshold.
        t_min (float): minimum track length in frames.

    Returns:
        list: list of tracks.
    """

    _check_types(sigma_l, sigma_h, sigma_iou, t_min, t_miss_max)

    print('Extended IOU')
    
    tracks_active: list = []
    tracks_finished = []
    vehID: int = 0
    vehIDs_finished: list = []
    new_detections: dict = {}

    for frame_num in tqdm(detections, desc="Tracked frames", unit="frames"):
        detections_frame = detections[frame_num][DETECTIONS]
        # apply low threshold to detections
        dets = [det for det in detections_frame if det[CONFIDENCE] >= sigma_l]
        new_detections[frame_num] = {}
        updated_tracks: list = []
        saved_tracks: list = []
        
        for track in tracks_active:
            direction_vector: list = []
            if dets: # check if dets in not empty
                # get det with highest iou
   
                best_match = max(
                    dets, key=lambda x: iou(track[BBOXES][-1], make_bbox(x))
                    )
                                                        
                direction_vector = get_direction_vector(track, best_match)
                
                
                if (track['mode'] == 'IOU' and iou(track[BBOXES][-1], make_bbox(best_match)) >= sigma_iou) or (track['mode'] == 'Extrapolated' and iou(track[BBOXES][-1], make_bbox(best_match)) >= (sigma_iou - 0.)): # or extrapolated BB >= sigma_iou (smaller cutoff?)
                    track[FRAMES].append(int(frame_num))
                    track[BBOXES].append(make_bbox(best_match))
                    track[CENTER].append(center(best_match))
                    track[CONFIDENCE].append(best_match[CONFIDENCE])
                    track[CLASS].append(best_match[CLASS])
                    track[MAX_CONF] = max(track[MAX_CONF], best_match[CONFIDENCE])
                    track[AGE] = 0
                    track[X].append(direction_vector[0])
                    track[Y].append(direction_vector[1])
                    track[W].append(best_match[W])
                    track[H].append(best_match[H])
                    track[X_VECTOR].append(direction_vector[0])
                    track[Y_VECTOR].append(direction_vector[1])
                    track['vector_rolling_mean_rotated'].append([direction_vector[0], direction_vector[1]])
                    # track[VECTOR_AMOUNT].append(direction_vector[2])
                    track['mode_list'].append('IOU')
                    
                    
                    # if vector[-1] close to vector: append
                    updated_tracks.append(track)

                    # remove best matching detection from detections
                    del dets[dets.index(best_match)]
                    # best_match    [TRACK_ID] = track[TRACK_ID]
                    best_match[FIRST] = False
                    new_detections[frame_num][track[TRACK_ID]] = best_match



            # if track was not updated
            if not updated_tracks or track is not updated_tracks[-1]:
                if track[AGE] <= 10 and len(track['frames']) >= 5: #t_extrapolate
                    if track[MAX_CLASS] == 'bicycle':
                        track['classmode_list'].append('bicycle')
                        track['rolling_mean_center'] = get_rolling_mean(first =  track[X_VECTOR][-5:],  second = track[Y_VECTOR][-5:])
                        track['rolling_mean_center_list'].append(track['rolling_mean_center'])
                        track[VECTOR_ROLLING_MEAN] = list((track['rolling_mean_center'][0], track['rolling_mean_center'][1]))
                        track['VECTOR_ROLLING_MEAN_list'].append(track[VECTOR_ROLLING_MEAN])
                        rolling_mean_BBOX = get_rolling_mean(first =  track[W][-5:],  second = track[H][-5:])
                        track[BBOXES_ROLLING_MEAN] = list((rolling_mean_BBOX[0], rolling_mean_BBOX[1]))
                        
                        # Extrapolate BB with vector rolling means
                        track[CENTER_EXTRAPOLATED] = [a + b for a, b in zip(list(track[CENTER][-1]), track[VECTOR_ROLLING_MEAN])]
                        track['CENTER_EXTRAPOLATED_list'].append(track[CENTER_EXTRAPOLATED])
                    
                    elif track[MAX_CLASS] == 'pedestrian':
                        track['classmode_list'].append('pedestrian')
                        track['rolling_mean_center'] = get_rolling_mean(first =  track[X_VECTOR][-5:],  second = track[Y_VECTOR][-5:])
                        track['rolling_mean_center_list'].append(track['rolling_mean_center'])
                        track[VECTOR_ROLLING_MEAN] = list((track['rolling_mean_center'][0], track['rolling_mean_center'][1]))
                        track['VECTOR_ROLLING_MEAN_list'].append(track[VECTOR_ROLLING_MEAN])
                        rolling_mean_BBOX = get_rolling_mean(first =  track[W][-5:],  second = track[H][-5:])
                        track[BBOXES_ROLLING_MEAN] = list((rolling_mean_BBOX[0], rolling_mean_BBOX[1]))
                        
                        # Extrapolate BB with vector rolling means
                        track[CENTER_EXTRAPOLATED] = [a + b for a, b in zip(list(track[CENTER][-1]), track[VECTOR_ROLLING_MEAN])]
                        track['CENTER_EXTRAPOLATED_list'].append(track[CENTER_EXTRAPOLATED])
                        
                        

                    else:
                        track['classmode_list'].append('other')
                        track['rolling_mean_center'] = get_rolling_mean(first =  track[X_VECTOR][-7:-2],  second = track[Y_VECTOR][-7:-2])
                        track['rolling_mean_center_list'].append(track['rolling_mean_center'])
                        if track['mode_list'][-1] == 'IOU': # factor in smaller vector of the last regular step because of partial Verdeckung                          
                            track[VECTOR_ROLLING_MEAN] = list((track['rolling_mean_center'][0], track['rolling_mean_center'][1]))
                            
                            def multi(vector):
                                return vector * 2
                            track[VECTOR_ROLLING_MEAN] = list(map(multi,track[VECTOR_ROLLING_MEAN]))
                        else:
                            track[VECTOR_ROLLING_MEAN] = list((track['rolling_mean_center'][0], track['rolling_mean_center'][1]))
                        track['VECTOR_ROLLING_MEAN_list'].append(track[VECTOR_ROLLING_MEAN])
                        rolling_mean_BBOX = get_rolling_mean(first =  track[W][-7:-2],  second = track[H][-7:-2])
                        track[BBOXES_ROLLING_MEAN] = list((rolling_mean_BBOX[0], rolling_mean_BBOX[1]))
                        
                        
                        # Extrapolate BB with vector rolling means
                        # Include rotation
                        track['rolling_mean_angle'].append(get_rolling_mean_angle(vectors = track['rolling_mean_center_list']))
                        track['vector_rolling_mean_rotated'].append(rotate(track[VECTOR_ROLLING_MEAN], track['rolling_mean_angle'][-1]))
                        track[CENTER_EXTRAPOLATED] = [a + b for a, b in zip(list(track[CENTER][-1]), track['vector_rolling_mean_rotated'][-1])]
                        track['CENTER_EXTRAPOLATED_list'].append(track[CENTER_EXTRAPOLATED])
                        track[X_VECTOR].append(track['vector_rolling_mean_rotated'][0][0])
                        track[Y_VECTOR].append(track['vector_rolling_mean_rotated'][0][1])

                        
                    track[BBOXES_EXTRAPOLATED] = tuple(make_extrapolated_bbox(track[CENTER_EXTRAPOLATED] + [track[BBOXES_ROLLING_MEAN][0], track[BBOXES_ROLLING_MEAN][1]]))
                    track[BBOXES].append(track[BBOXES_EXTRAPOLATED])
                    track[X].append(track[CENTER_EXTRAPOLATED][0])
                    track[Y].append(track[CENTER_EXTRAPOLATED][1])
                    track[W].append(track[BBOXES_ROLLING_MEAN][0])
                    track[H].append(track[BBOXES_ROLLING_MEAN][1])
                    
                    track[CENTER].append(tuple(track[CENTER_EXTRAPOLATED]))
                    track[FRAMES].append(int(frame_num))
                    track['mode_list'].append('Extrapolated')
                    track[CONFIDENCE].append(track[CONFIDENCE][-1])
                    track[CLASS].append(track[CLASS][-1])
                    
                    # # Include tails of tracks not picked up again
                    # temp = {}
                    # temp[CLASS] = track[MAX_CLASS]
                    # temp[CONFIDENCE] = track[MAX_CONF]
                    # temp[X] = track[CENTER_EXTRAPOLATED][0]
                    # temp[Y] = track[CENTER_EXTRAPOLATED][1]
                    # temp[W] = track[BBOXES_ROLLING_MEAN][0]
                    # temp[H] = track[BBOXES_ROLLING_MEAN][1]
                    # temp['frame'] = best_match['frame']
                    # temp['occurrence'] = best_match['occurrence']
                    # temp['input_file_path'] = best_match['input_file_path']
                    # temp['interpolated-detection'] = best_match['interpolated-detection']
                    
                    # new_detections[frame_num][track[TRACK_ID]] = temp

                # finish track when the conditions are met
                if track[AGE] < t_miss_max:
                    track[AGE] += 1
                    saved_tracks.append(track)
                elif (
                    track[MAX_CONF] >= sigma_h
                    and track[FRAMES][-1] - track[FRAMES][0] >= t_min
                ):
                    tracks_finished.append(track)
                    vehIDs_finished.append(track[TRACK_ID])
        # TODO: Alter der Tracks
        # create new tracks
        new_tracks = []
        for det in dets:
            vehID += 1
            new_tracks.append(
                {
                    FRAMES: [int(frame_num)],
                    BBOXES: [make_bbox(det)],
                    CENTER: [center(det)],
                    CONFIDENCE: [det[CONFIDENCE]],
                    CLASS: [det[CLASS]],
                    MAX_CLASS: det[CLASS],
                    MAX_CONF: det[CONFIDENCE],
                    TRACK_ID: vehID,
                    START_FRAME: int(frame_num),
                    AGE: 0,
                    X_VECTOR: [],
                    Y_VECTOR: [],
                    W: [],
                    H: [],
                    # VECTOR_AMOUNT: [],
                    VECTOR_ROLLING_MEAN: [],
                    CENTER_EXTRAPOLATED: [],
                    BBOXES_EXTRAPOLATED: [],
                    DIMENSIONS: [],
                    BBOXES_ROLLING_MEAN: [],
                    'CENTER_EXTRAPOLATED_list': [],
                    MODE: 'IOU',
                    'mode_list': [],
                    X: [],
                    Y: [],
                    'VECTOR_ROLLING_MEAN_list': [],
                    'rolling_mean_center_list': [],
                    'rolling_mean_angle': [],
                    'vector_rolling_mean_rotated': [],
                    'classmode_list': []
                    
                }
            )
            # det[TRACK_ID] = vehID
            det[FIRST] = True
            new_detections[frame_num][vehID] = det
        tracks_active = updated_tracks + saved_tracks + new_tracks

    # finish all remaining active tracks
    # tracks_finished += [
    #     track
    #     for track in tracks_active
    #     if (
    #         track["max_conf"] >= sigma_h
    #         and track["frames"][-1] - track["frames"][0] >= t_min
    #     )
    # ]

    # for track in tracks_finished:
    #     track["max_class"] = pd.Series(track["class"]).mode().iat[0]

    # TODO: #82 Use dict comprehensions in track_iou
    for frame_det in tqdm(
        new_detections.values(), desc="New detection frames", unit="frames"
    ):
        for vehID, det in frame_det.items():
            det[FINISHED] = vehID in vehIDs_finished
            det[TRACK_ID] = vehID
    # return tracks_finished
    # TODO: #83 Remove unnessecary code (e.g. for tracks_finished) from track_iou
    
    
    return new_detections


def _check_types(
    sigma_l: float, sigma_h: float, sigma_iou: float, t_min: int, t_miss_max: int
) -> None:
    """Raise ValueErrors if wrong types"""

    if not isinstance(sigma_l, (int, float)):
        raise ValueError("sigma_l has to be int or float")
    if not isinstance(sigma_h, (int, float)):
        raise ValueError("sigma_h has to be int or float")
    if not isinstance(sigma_iou, (int, float)):
        raise ValueError("sigma_iou has to be int or float")
    if not isinstance(t_min, int):
        raise ValueError("t_min has to be int")
    if not isinstance(t_miss_max, int):
        raise ValueError("t_miss_max has to be int")
