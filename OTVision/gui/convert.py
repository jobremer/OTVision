# OTVision: Python gui to convert h264 based videos to other formats and frame rates

# Copyright (C) 2020 OpenTrafficCam Contributors
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


import PySimpleGUI as sg
from gui.helpers.frames import OTFrameFoldersFiles
from gui.helpers.texts import OTTextSpacer
from gui.helpers.sg_otc_theme import (
    OTC_ICON,
    OTC_THEME,
)
from config import CONFIG
from convert.convert import convert
from helpers.files import get_files


def main(sg_theme=OTC_THEME):
    folders = CONFIG["LAST PATHS"]["FOLDERS"]
    single_files = CONFIG["LAST PATHS"]["VIDEOS"]
    files = get_files(
        paths=[*folders, *single_files],
        filetypes=[*CONFIG["FILETYPES"]["VID"], ".h264"],
    )
    sg.SetOptions(font=(CONFIG["GUI"]["FONT"], CONFIG["GUI"]["FONTSIZE"]))

    # Get initial layout and create initial window
    layout, frame_folders_files = create_layout(files)
    window = sg.Window(
        title="OTVision: Convert",
        layout=layout,
        icon=OTC_ICON,
        location=(
            CONFIG["GUI"]["WINDOW"]["LOCATION_X"],
            CONFIG["GUI"]["WINDOW"]["LOCATION_Y"],
        ),
        element_justification="center",
        resizable=True,
        finalize=True,
    )
    window.maximize()
    frame_folders_files.listbox_files.expand(expand_x=True)
    window["-progress_convert-"].update_bar(0)

    # Event Loop to process "events" and get the "values" of the inputs
    while True:
        event, values = window.read()
        print(event)
        # Close Gui
        if event in [sg.WIN_CLOSED, "Cancel", "-BUTTONBACKTOHOME-"]:
            break
        # Set parameters (no actions required for now)
        elif event == "-drop_output_filetype-":
            pass
        elif event == "-check_fps_from_filename-":
            pass
        elif event == "-in_input_fps-":
            pass
        elif event == "-in_output_fps-":
            pass
        elif event == "-check_overwrite-":
            pass
        elif event == "-button_convert-":
            for i, file in enumerate(files):
                convert(
                    input_video=file,
                    output_filetype=values["-drop_output_filetype-"],
                    input_fps=values["-in_input_fps-"],
                    # output_fps=values["-in_output_fps-"],
                    fps_from_filename=values["-check_fps_from_filename-"],
                    overwrite=values["-check_overwrite-"],
                )
                window["-progress_convert-"].update(current_count=i + 1, max=len(files))
        # Folders and files
        files = frame_folders_files.process_events(event, values, files)
        window["-progress_convert-"].update(current_count=0, max=len(files))

    window.close()


def create_layout(files):

    # GUI elements: Choose videos
    vid_filetypes = CONFIG["FILETYPES"]["VID"]
    vid_filetypes.append(".h264")
    frame_folders_files = OTFrameFoldersFiles(filetype=".h264")

    # GUI elements: Set parameters
    width_c1 = int(CONFIG["GUI"]["FRAMEWIDTH"] / 2)
    width_c2 = 5
    text_output_filetype = sg.T(
        "Output filetype", justification="right", size=(width_c1, 1)
    )
    drop_output_filetype = sg.Drop(
        [*CONFIG["FILETYPES"]["VID"]],
        default_value=CONFIG["CONVERT"]["OUTPUT_FILETYPE"],
        key="-drop_output_filetype-",
    )
    text_fps_from_filename = sg.T(
        "Try to use framerate from input video",
        justification="right",
        size=(width_c1, 1),
    )
    check_fps_from_filename = sg.Check(
        "", default=True, key="-check_fps_from_filename-"
    )
    text_input_fps = sg.T("Input framerate", justification="right", size=(width_c1, 1))
    in_input_fps = sg.In(
        CONFIG["CONVERT"]["FPS"],
        key="-in_input_fps-",
        enable_events=True,
        size=(width_c2, 10),
    )
    text_output_fps = sg.T(
        "Output framerate", justification="right", size=(width_c1, 1)
    )
    in_output_fps = sg.In(
        CONFIG["CONVERT"]["FPS"],
        key="-in_output_fps-",
        enable_events=True,
        size=(width_c2, 10),
    )
    text_overwrite = sg.T(
        "Overwrite existing videos",
        justification="right",
        size=(width_c1, 1),
    )
    check_overwrite = sg.Check(
        "", default=CONFIG["CONVERT"]["OVERWRITE"], key="-check_overwrite-"
    )
    frame_parameters = sg.Frame(
        "Step 2: Set parameters",
        [
            [OTTextSpacer()],
            [text_output_filetype, drop_output_filetype],
            [text_fps_from_filename, check_fps_from_filename],
            [text_input_fps, in_input_fps],
            # [text_output_fps, in_output_fps],
            [text_overwrite, check_overwrite],
            [OTTextSpacer()],
        ],
        size=(100, 10),
    )

    # Gui elements: Convert
    button_convert = sg.B("Convert!", key="-button_convert-")
    progress_convert = sg.ProgressBar(
        max_value=len(files),
        size=(CONFIG["GUI"]["FRAMEWIDTH"] / 2, 20),
        # bar_color=("red", "green"),
        key="-progress_convert-",
    )
    frame_convert = sg.Frame(
        "Step 3: Start conversion",
        [
            [OTTextSpacer()],
            [button_convert],
            [progress_convert],
            [OTTextSpacer()],
        ],
        element_justification="center",
    )

    # Put layout together
    col_all = sg.Column(
        [[frame_folders_files], [frame_parameters], [frame_convert]],
        scrollable=True,
        expand_y=True,
    )
    layout = [[col_all]]

    return layout, frame_folders_files
