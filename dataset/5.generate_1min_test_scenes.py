#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import subprocess
from pathlib import Path
from tqdm import tqdm

DATASET_DIR = Path('/media/fharookshaik/DATA/mini_project_ii/dataset')
TRAINING_DIR = os.path.join(DATASET_DIR,'Test')
TRAINING_VIDEOS_DIR = os.path.join(TRAINING_DIR,'videos')
TRAINING_SCENES_DIR = os.path.join(TRAINING_DIR,'scenes')

def split_video(file):
    input_file= os.path.join(TRAINING_VIDEOS_DIR,file)
    output_dir= os.path.join(TRAINING_SCENES_DIR,file.split('.')[0])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    result = subprocess.run(["ffmpeg", "-i", input_file,"-map","0", "-c", "copy", "-f", "segment", "-segment_time", "60","-reset_timestamps", "1", output_dir+"/"+file.split(".")[0]+".%02d."+file.split(".")[-1]],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)


videos = os.listdir(TRAINING_VIDEOS_DIR)    
pbar = tqdm(videos)

for video in pbar:
    pbar.set_description(f'Generating Scenes {video}')
    split_video(video)

# for idx,video in tqdm(enumerate(videos[:10]),total=len(videos[:10]),desc='Extracting'):
#     # print(video)
#     split_video(video)