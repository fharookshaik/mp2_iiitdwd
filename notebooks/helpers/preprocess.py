# Imports
import os
import pandas as pd
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math

import tensorflow as tf
from sklearn.utils import shuffle

# local imports
from helpers.config import DatasetPaths

data_paths = DatasetPaths()
# print(data_paths.PROJECT_DIR)

# Global Variables
IMG_SIZE = data_paths.IMG_SIZE
BATCH_SIZE = data_paths.BATCH_SIZE
MAX_FRAMES = data_paths.MAX_FRAMES
IMG_BATCH_SIZE = data_paths.IMG_BATCH_SIZE

# DATASET Paths
PROJECT_DIR = data_paths.PROJECT_DIR
DATASET_DIR = data_paths.DATASET_DIR
TRAINING_DIR = data_paths.TRAINING_DIR
TRAINING_DATASET_DIR = data_paths.TRAINING_DATASET_DIR

TEST_DIR = data_paths.TEST_DIR
TEST_DATASET_DIR = data_paths.TEST_DATASET_DIR

TRAIN_CSV = data_paths.TRAIN_CSV
TEST_CSV = data_paths.TEST_CSV

VALIDATION_CSV = data_paths.VALIDATION_CSV


class Data:
    def __init__(self,csv_file,dataset_dir):
        self.csv_file = csv_file
        self.dataset_dir = dataset_dir
        self.data = pd.read_csv(csv_file)
    
        self.remove_missing_data_csv()

    def get_df(self):
        return self.data

    # remove missing data from csv
    def remove_missing_data_csv(self):
        # Removing missing data from csv
        print('Removing Missing Data from csv')
        for index,val in self.data.iterrows():
            if val['Scene_ID'] < 10:
                filename = f"{val['Video ID']}.0{val['Scene_ID']}.mp4"
            else:
                filename = f"{val['Video ID']}.{val['Scene_ID']}.mp4"

            filePath = os.path.join(self.dataset_dir,filename)
            if not os.path.exists(filePath):
                print(f'{filename[:-4]} not found. Removing entry from data')
                self.data.drop(index,inplace=True)

        print('Final dataset shape: ', self.data.shape)

    # verify frame rate
    def verify_frame_rate(self):        
        # Verifying Frame Rate
        errorFrameRate = []

        for index,val in self.data.iterrows():
            if val['Scene_ID'] < 10:
                filename = f"{val['Video ID']}.0{val['Scene_ID']}.mp4"
            else:
                filename = f"{val['Video ID']}.{val['Scene_ID']}.mp4"

            filePath = os.path.join(self.dataset_dir,filename)
            # Getting frameRate
            frameRate = cv.VideoCapture(filePath).get(cv.CAP_PROP_FPS)
            if round(frameRate,2) != round(val['Original Video Avg Framerate'],2):
                errorFrameRate.append(filename[:-4])
                print('Frame Rate Not Matched with actual : {}'.format(filename))
                print('FrameRate from Video: {} | FramRate from data: {}'.format(round(frameRate,2),round(val['Original Video Avg Framerate'],2)))
                print('-'*20)

        print('Total error detected framerate files : ',len(frameRate))

class VideoDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,df,dataset_dir='',batch_size=BATCH_SIZE,max_frames=MAX_FRAMES,shuffle=True,only_one_video=False) -> None:
        self.df = df
        self.batch_size= batch_size
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir
        self.only_one_video = only_one_video

        if self.only_one_video:
            self.batch_size = 1

        
        self.samples = []
        self.df['Video_Scene ID'] = self.df['Video ID'].astype(str) + '.' + self.df['Scene_ID'].astype(str)
        try:
            for idx,val in self.df.iterrows():
                if val['Scene_ID'] < 10:
                    filename = f"{val['Video ID']}.0{val['Scene_ID']}.mp4"
                else:
                    filename = f"{val['Video ID']}.{val['Scene_ID']}.mp4"

                Id = val['Video_Scene ID']
                filePath = os.path.join(self.dataset_dir,filename)
                label = val['Presence of Comic Mischief Content in Scene']


                if os.path.exists(filePath):
                    # Adding (id,filepath,label) to samples
                    self.samples.append([Id,filePath,label]) 
                else:
                    raise FileNotFoundError(f'{filePath} Not Found')

        except Exception as e:
            print('Error generating samples : ',e)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.samples)/self.batch_size)

    def __getitem__(self,index):
        total_samples = len(self.samples)
        # generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_samples_temp = [self.samples[k] for k in indexes]
        X,y = self._preprocess_data(list_samples_temp)

        return X,y

    def _crop_center_square(self,frame):
        y,x = frame.shape[0:2]
        min_dim = min(x,y)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x: start_x + min_dim]
    
    # def _load_video(self,path):
    #     frames = []
    #     cap = cv.VideoCapture(path)
    #     frameRate = cap.get(cv.CAP_PROP_FPS)
    #     frameCount = cap.get(cv.CAP_PROP_FRAME_COUNT)
    
    #     tempSplit = frameCount // MAX_FRAMES
    #     count = 0
    #     try:
    #         while cap.isOpened():
    #             frameId = cap.get(1)
    #             ret, frame = cap.read()

    #             if not ret:
    #                 break
                
    #             if frameId % tempSplit == 0:
    #                 frame = self._crop_center_square(frame)
    #                 frame = cv.resize(frame, (IMG_SIZE,IMG_SIZE))
    #                 frame = frame[:,:,[2,1,0]]
    #                 frames.append(frame)
    #                 count += 1

    #             #  restricting to extract only 60 frames
    #             if count >= MAX_FRAMES:
    #                 break
            
    #         # Adding empty frames for videos lesser than 60 frames
    #         if count <= MAX_FRAMES:
    #             frame = np.zeros((IMG_SIZE,IMG_SIZE,3))
    #             for _ in range(MAX_FRAMES-count):
    #                 frames.append(frame)

    #         frames = np.array(frames)
    #         frames = frames / 255

    #     finally:
    #         cap.release()
        
    #     return frames

    def _load_video(self,path):
        frames = []
        cap = cv.VideoCapture(path)
        frameRate = cap.get(cv.CAP_PROP_FPS)
        count = 0
        try:
            while cap.isOpened():
                frameId = cap.get(1)
                ret, frame = cap.read()

                if not ret:
                    break
                
                if frameId % math.floor(frameRate) == 0:
                    frame = self._crop_center_square(frame)
                    frame = cv.resize(frame, (IMG_SIZE,IMG_SIZE))
                    frame = frame[:,:,[2,1,0]]
                    frames.append(frame)
                    count += 1

                #  restricting to extract only 60 frames
                if count >= MAX_FRAMES:
                    break
            
            # Adding empty frames for videos lesser than 60 frames
            if count <= MAX_FRAMES:
                frame = np.zeros((IMG_SIZE,IMG_SIZE,3))
                for _ in range(MAX_FRAMES-count):
                    frames.append(frame)

            frames = np.array(frames)
            frames = frames / 255

        finally:
            cap.release()
        
        return frames


    def _preprocess_data(self,samples):
        if self.only_one_video:
            Id,filepath,label = samples[0]
            X = self._load_video(filepath)
            y = label
        else:    
            X = []
            y = []
            for idx,val in enumerate(samples):
                Id,filepath,label = val
                X.append(self._load_video(filepath))
                y.append(label)
        X = np.array(X)
        y = np.array(y).reshape((-1,1))
        return X,y        

# class ImageDataGenerator():
#     def __init__(self,df,dataset_dir='',batch_size=BATCH_SIZE,max_frames=MAX_FRAMES,shuffle=True,only_one_video=False) -> None:
#         self.df = df
#         self.batch_size= batch_size
#         self.max_frames = max_frames
#         self.shuffle = shuffle
#         self.dataset_dir = dataset_dir

        
#         self.samples = []
#         self.df['Video_Scene ID'] = self.df['Video ID'].astype(str) + '.' + self.df['Scene_ID'].astype(str)
#         try:
#             for idx,val in self.df.iterrows():
#                 if val['Scene_ID'] < 10:
#                     filename = f"{val['Video ID']}.0{val['Scene_ID']}.mp4"
#                 else:
#                     filename = f"{val['Video ID']}.{val['Scene_ID']}.mp4"

#                 Id = val['Video_Scene ID']
#                 filePath = os.path.join(self.dataset_dir,filename)
#                 label = val['Presence of Comic Mischief Content in Scene']


#                 if os.path.exists(filePath):
#                     # Adding (id,filepath,label) to samples
#                     self.samples.append([Id,filePath,label]) 
#                 else:
#                     raise FileNotFoundError(f'{filePath} Not Found')

#         except Exception as e:
#             print('Error generating samples : ',e)

#         self.on_epoch_end()

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.samples))
#         if self.shuffle == True:
#             np.random.shuffle(self.indexes)

#     def __len__(self):
#         return math.ceil(len(self.samples)/self.batch_size)

#     def __getitem__(self,index):
#         total_samples = len(self.samples)
#         # generates indexes of the batch
#         indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

#         list_samples_temp = [self.samples[k] for k in indexes]
#         X,y = self._preprocess_data(list_samples_temp)

#         return X,y

#     def _crop_center_square(self,frame):
#         y,x = frame.shape[0:2]
#         min_dim = min(x,y)
#         start_x = (x // 2) - (min_dim // 2)
#         start_y = (y // 2) - (min_dim // 2)
#         return frame[start_y : start_y + min_dim, start_x: start_x + min_dim]
    
#     def _load_video(self,path):
#         frames = []
#         cap = cv.VideoCapture(path)
#         frameRate = cap.get(cv.CAP_PROP_FPS)
#         count = 0
#         try:
#             while cap.isOpened():
#                 frameId = cap.get(1)
#                 ret, frame = cap.read()

#                 if not ret:
#                     break
                
#                 if frameId % math.floor(frameRate) == 0:
#                     frame = self._crop_center_square(frame)
#                     frame = cv.resize(frame, (IMG_SIZE,IMG_SIZE))
#                     frame = frame[:,:,[2,1,0]]
#                     frames.append(frame)
#                     count += 1

#                 #  restricting to extract only 60 frames
#                 if count >= MAX_FRAMES:
#                     break
            
#             # Adding empty frames for videos lesser than 60 frames
#             if count <= MAX_FRAMES:
#                 frame = np.zeros((IMG_SIZE,IMG_SIZE,3))
#                 for _ in range(MAX_FRAMES-count):
#                     frames.append(frame)

#             frames = np.array(frames)
#             frames = frames / 255

#         finally:
#             cap.release()
        
#         return frames

#     def _preprocess_data(self,samples):
#         if self.only_one_video:
#             Id,filepath,label = samples[0]
#             X = self._load_video(filepath)
#             y = label
#         else:    
#             X = []
#             y = []
#             for idx,val in enumerate(samples):
#                 Id,filepath,label = val
#                 X.append(self._load_video(filepath))
#                 y.append(label)
#         X = np.array(X)
#         y = np.array(y).reshape((-1,1))
#         return X,y        


class ImageDataGenerator(tf.keras.utils.Sequence):
    def __init__(self,df,dataset_dir='',batch_size=IMG_BATCH_SIZE,max_frames=MAX_FRAMES,shuffle=True) -> None:
        self.df = df
        self.batch_size= batch_size
        self.max_frames = max_frames
        self.shuffle = shuffle
        self.dataset_dir = dataset_dir

        
        self.samples = []
        self.df['Video_Scene ID'] = self.df['Video ID'].astype(str) + '.' + self.df['Scene_ID'].astype(str)
        try:
            for idx,val in self.df.iterrows():
                if val['Scene_ID'] < 10:
                    filename = f"{val['Video ID']}.0{val['Scene_ID']}.mp4"
                else:
                    filename = f"{val['Video ID']}.{val['Scene_ID']}.mp4"

                Id = val['Video_Scene ID']
                filePath = os.path.join(self.dataset_dir,filename)
                label = val['Presence of Comic Mischief Content in Scene']


                if os.path.exists(filePath):
                    # Adding (id,filepath,label) to samples
                    self.samples.append([Id,filePath,label]) 
                else:
                    raise FileNotFoundError(f'{filePath} Not Found')

        except Exception as e:
            print('Error generating samples : ',e)

        self.on_epoch_end()

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.samples))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return math.ceil(len(self.samples)/self.batch_size)

    def __getitem__(self,index):
        total_samples = len(self.samples)
        # generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        list_samples_temp = [self.samples[k] for k in indexes]
        X,y = self._preprocess_data(list_samples_temp)

        return X,y

    def _crop_center_square(self,frame):
        y,x = frame.shape[0:2]
        min_dim = min(x,y)
        start_x = (x // 2) - (min_dim // 2)
        start_y = (y // 2) - (min_dim // 2)
        return frame[start_y : start_y + min_dim, start_x: start_x + min_dim]
    
    def _load_video(self,path):
        frames = []
        cap = cv.VideoCapture(path)
        frameRate = cap.get(cv.CAP_PROP_FPS)
        count = 0
        try:
            while cap.isOpened():
                frameId = cap.get(1)
                ret, frame = cap.read()

                if not ret:
                    break
                
                if frameId % math.floor(frameRate) == 0:
                    frame = self._crop_center_square(frame)
                    frame = cv.resize(frame, (IMG_SIZE,IMG_SIZE))
                    frame = frame[:,:,[2,1,0]]
                    frames.append(frame)
                    count += 1

                #  restricting to extract only 60 frames
                if count >= MAX_FRAMES:
                    break
            
            # Adding empty frames for videos lesser than 60 frames
            if count <= MAX_FRAMES:
                frame = np.zeros((IMG_SIZE,IMG_SIZE,3))
                for _ in range(MAX_FRAMES-count):
                    frames.append(frame)

            frames = np.array(frames)
            frames = frames / 255

        finally:
            cap.release()
        
        return frames

    def _preprocess_data(self,samples):
        X = []
        y = []
        for idx,val in enumerate(samples):
            Id,filepath,label = val

            vid_frames = self._load_video(filepath)
            for frame in vid_frames:
                X.append(frame)
                y.append(label)
            
            # X.append(self._load_video(filepath))
            # y.append(label)
        X = np.array(X)
        y = np.array(y).reshape((-1,1))
        return X,y


#############################

def get_train_datagen(only_one_video=False,small_sample=False):
    data = Data(TRAIN_CSV,TRAINING_DATASET_DIR)
    data = data.get_df()
    
    if small_sample:
        print('Taking only 100 samples (50 true + 50 false)')
        grouped_data = data.groupby(['Presence of Comic Mischief Content in Scene'])
        true_df = grouped_data.get_group(1)
        false_df = grouped_data.get_group(0)

        if true_df.shape[0] > false_df.shape[0]:
            num = false_df.shape[0]
        else:
            num = true_df.shape[0]

        data = true_df.iloc[:num,:].append(false_df.iloc[:num,:])
        print('Final Dataframe shape : ',data.shape)

    data = shuffle(data)

    return VideoDataGenerator(data,TRAINING_DATASET_DIR,only_one_video=only_one_video)

def get_validation_datagen(only_one_video=False):
    data = Data(VALIDATION_CSV,TRAINING_DATASET_DIR)
    data = data.get_df()

    data = shuffle(data)

    return VideoDataGenerator(data,TRAINING_DATASET_DIR,only_one_video=only_one_video)

def get_img_datagen(batch_size=IMG_BATCH_SIZE, small_sample=False):
    data = Data(TRAIN_CSV,TRAINING_DATASET_DIR)
    data = data.get_df()

    if small_sample:
        grouped_data = data.groupby(['Presence of Comic Mischief Content in Scene'])
        true_df = grouped_data.get_group(1)
        false_df = grouped_data.get_group(0)

        if true_df.shape[0] < false_df.shape[0]:
            num = true_df.shape[0]
        else:
            num = false_df.shape[0]
        
        print(f'Taking only {num*2} samples ({num} true + {num} false)')
        
        data = true_df.iloc[:num,:].append(false_df.iloc[:num,:])
        print('Final Dataframe shape : ',data.shape)

    data = shuffle(data)

    # # Splitting the data in train, valid
    # ID_temp = np.arange(data.shape[0])
    return ImageDataGenerator(data,TRAINING_DATASET_DIR,batch_size=batch_size)


