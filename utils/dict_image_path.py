from glob import glob
import os 
import json

def create_dict_image_path(data_dir):
    dict_json_path = {}
    for folder in sorted(os.listdir(data_dir)):
        dict_json_path[folder] = dict()
        folder_path = os.path.join(data_dir, folder)
        for video_folder in sorted(os.listdir(folder_path)):
            dict_json_path[folder][video_folder] = dict()
            video_folder_path = os.path.join(folder_path, video_folder)
            for frame in sorted(os.listdir(video_folder_path)):
                dict_json_path[folder][video_folder]