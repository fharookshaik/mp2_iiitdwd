import os
from pathlib import Path

class DatasetPaths:
    # Global Variables
    IMG_SIZE = 224
    BATCH_SIZE = 1
    MAX_FRAMES = 60
    IMG_BATCH_SIZE = 2

    # DATASET Paths
    PROJECT_DIR = Path('/content/drive/MyDrive/Mini_Project_II')
    DATASET_DIR = os.path.join(PROJECT_DIR,'dataset')
    TRAINING_DIR = os.path.join(DATASET_DIR,'Training')
    TRAINING_DATASET_DIR = os.path.join(TRAINING_DIR,'training_data')

    TEST_DIR = os.path.join(DATASET_DIR,'Test')
    TEST_DATASET_DIR = os.path.join(TEST_DIR,'test_data')

    TRAIN_CSV = os.path.join(DATASET_DIR,'Dataset_ComicMischief_Training_Scene_Binary_Annotations.csv')
    TEST_CSV = os.path.join(DATASET_DIR,'Dataset_ComicMischief_Test_Scenes.csv')

    VALIDATION_CSV = os.path.join(DATASET_DIR,'Dataset_ComicMischief_Validation_Scene_Binary_Annotations.csv')