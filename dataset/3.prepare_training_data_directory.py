#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import pandas as pd
from pathlib import Path

DATASET_DIR = Path('/media/fharookshaik/DATA/mini_project_ii/dataset')
TRAINING_DIR = os.path.join(DATASET_DIR,'Training')
TRAINING_VIDEOS_DIR = os.path.join(TRAINING_DIR,'videos')
TRAINING_SCENES_DIR = os.path.join(TRAINING_DIR,'scenes')
TRAINING_DATA = os.path.join(TRAINING_DIR,'training_data')


scenesDirectory=f"{TRAINING_SCENES_DIR}/"
trainingDataDirectory=f"{TRAINING_DATA}/"

if not os.path.exists(trainingDataDirectory):
    os.makedirs(trainingDataDirectory)
    
csvFile=pd.read_csv('/media/fharookshaik/DATA/mini_project_ii/dataset/Dataset_ComicMischief_Training_Scene_Binary_Annotations.csv')  

totalSuccessfullAttempts=0
totalFailedAttempts=0
for i,videoID in enumerate(csvFile["Video ID"]):
    try:
        sceneID = str(csvFile["Scene_ID"][i])
    
        shutil.copyfile(scenesDirectory+videoID+"/"+videoID+".0"+sceneID+".mp4", trainingDataDirectory+videoID+".0"+sceneID+".mp4")
        print("Video ID:",videoID,"Scene ID:",sceneID, "Successfully extracted")
        totalSuccessfullAttempts+=1
    except Exception as e:
        print("Video ID:",videoID,"Scene ID:",sceneID, "Failed. Exception: ", e)
        totalFailedAttempts+=1
print("Total Successfull Scene Extraction Attempts = ", totalSuccessfullAttempts)
print("Total Failed Scene Extraction Attempts = ", totalFailedAttempts)

    
