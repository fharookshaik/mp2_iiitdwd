# pip install pytube==10.4.1
# pip install -U get-video-properties
import csv
import pytube
import os
import pandas as pd
import urllib.request
from videoprops import get_video_properties
from pathlib import Path

DATASET_DIR = Path('/media/fharookshaik/DATA/mini_project_ii/dataset')
TRAINING_DIR = os.path.join(DATASET_DIR,'Test')
TRAINING_VIDEOS_DIR = os.path.join(TRAINING_DIR,'videos')

def downloadYoutube(filename,url):
    try:
        from pathlib import Path
        if not os.path.exists(Path(TRAINING_VIDEOS_DIR,'.'.join([filename,'mp4']))):
            yt = pytube.YouTube(url)
            video = yt.streams.get_highest_resolution()
            out_file = video.download(f'{TRAINING_VIDEOS_DIR}/')
            ext = out_file.split('.')[-1]
            yt_id_filename = '.'.join([filename,ext])
            os.rename(out_file,f'{TRAINING_VIDEOS_DIR}/'+yt_id_filename)
            print(yt_id_filename + " download successful")
        else:
            print('File already found.Skipping {0}.mp4'.format(filename))
    except Exception as e:
        print('Error downloading {0} : {1}'.format(filename,e))

def downloadIMDB(filename,url):
    from pathlib import Path
    if not os.path.exists(Path(TRAINING_VIDEOS_DIR,'.'.join([filename,'mp4']))):
        videoplayerlink=url
        html=""
        with urllib.request.urlopen(videoplayerlink) as response:
            html = response.read()
        html=str(html)
        script = html.split('"video/mp4')
        videoLink=script[1].split('},{')[0].split('"')[4][0:-2]

        urllib.request.urlretrieve(videoLink,os.path.join(TRAINING_VIDEOS_DIR,f'{filename}.mp4'))
    else:
        print('File Already Found.Skipping {0}.mp4'.format(filename)) 


    
if not os.path.exists(TRAINING_VIDEOS_DIR):
    os.mkdir(TRAINING_VIDEOS_DIR)
final_videos=[]

csvFile=pd.read_csv('/media/fharookshaik/DATA/mini_project_ii/dataset/Dataset_ComicMischief_Test_Scenes.csv')  

def compute_frame_rate_from_string_fraction(fraction):
    num,den = fraction.split( '/' )
    result= float(num)/float(den)
    return result

#### Main Script - Downloading Videos from both Youtube and IMDB
import time

#### Main Script - Downloading Videos from both Youtube and IMDB
totalSuccessfullAttempts=0
missingVideos=[]
for i,URL in enumerate(csvFile["Video URL"]):
    # time.sleep(0.05)
    if(i==0 or (csvFile["Video URL"][i-1] not in URL)):
        try:
            if not os.path.exists(TRAINING_VIDEOS_DIR+f'{csvFile["Video ID"][i]}.mp4'):
                if "youtube" in URL:
                    downloadYoutube(csvFile["Video ID"][i],URL)
            
                if "imdb" in URL:
                    downloadIMDB(csvFile["Video ID"][i],URL)
            else:
                print('File already found. Skipping {0}.mp4'.format(csvFile["Video ID"][i]))
                
            props = get_video_properties(os.path.join(TRAINING_VIDEOS_DIR,f"{csvFile['Video ID'][i]}.mp4"))
            print(f"{i}. Video ID:" +"\t"+csvFile["Video ID"][i] +"\t URL: \t"+ URL+"\t"+"Codec:"+"\t"+  props['codec_name']+"\t"+ "Resolution:"+"\t"+ str(props['width']),"x",str(props['height']) +"\t"+ "Avg Frame rate:" +"\t"+ str(compute_frame_rate_from_string_fraction(props['avg_frame_rate'])))
            totalSuccessfullAttempts+=1
        except Exception as e:
            print(f"{i}. Video ID: " +"\t"+csvFile["Video ID"][i] +"\t URL: \t"+ URL+"\t"+ "Unable to download video")
            print('Exception : ',e)
            missingVideos.append((csvFile['Video ID'][i],URL))
print("Total Successfull Attempts = ", totalSuccessfullAttempts)
print("Missing Videos = ", missingVideos)