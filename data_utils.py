import librosa
from numba.core import config
import numpy as np, pandas as pd
import json
import matplotlib.pyplot as plt
import os, shutil
import sqlite3
import cv2
import config


def move_files(source_dir,target_dir):
    file_names = os.listdir(source_dir)
        
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)

def extract_feat(y,sr):
    # extract the various features of the audio
    mfcc = librosa.feature.mfcc(y = y, sr = sr, n_mfcc=13).T
    mfcc1stDel = librosa.feature.delta(mfcc)
    mfcc2ndDel = librosa.feature.delta(mfcc,order=2)
    mfcc = np.concatenate((mfcc,mfcc1stDel,mfcc2ndDel),axis=1)
    
    mfcc = np.mean(mfcc, axis = 0)  
    mel = np.mean(librosa.feature.melspectrogram(y = y, sr = sr, n_fft=2048).T, axis = 0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S = stft, y = y, sr = sr).T, axis = 0)
    contrast = np.mean(librosa.feature.spectral_contrast(S = stft, y = y, sr = sr).T, axis = 0)
    tonnetz =  np.mean(librosa.feature.tonnetz(y = librosa.effects.harmonic(y), sr = sr).T, axis = 0)

    return mfcc,chroma,mel,contrast,tonnetz # shape: (40,), (12,), (128,), (7,), (6,)

def plot_audio(y,sr):
    plt.figure(figsize=(12, 4))
    librosa.display.waveplot(y, sr=sr)
    plt.show()

def plot_spectrogram(y,sr):
    X = librosa.stft(y.astype('float'))
    Xdb = librosa.amplitude_to_db(X)
    plt.figure(figsize=(12, 5))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
    plt.show()

def remove_silence(y):
    y, _ = librosa.effects.trim(y)
    return y

def speed_aug(y,low=0.9,high=1.4):
    y_speed = y.copy()
    speed_change = np.random.uniform(low,high)
    tmp = librosa.effects.time_stretch(y_speed.astype('float64'), speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0 
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def volume_aug(y,low=0.5,high=5):
    y_aug = y.copy()
    dyn_change = np.random.uniform(low,high)
    y_aug = y_aug * dyn_change
#     print(y_aug[:50])
    return y_aug

def noise_aug(y,low=0.03,high=0.09):
    for i in np.random.randint(0,len(y),len(y)//800):
        val = np.random.uniform(low,high)
        y[i] += y[i] + val
    return y

def load_json_data(path):
    with open(path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    f.close()
    return json_data

def write_json_data(path,data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    f.close()

def addDataToDB(empId,fullName,emailID):
    print(f"{emailID},{empId},{fullName}")
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    # data = {
    #     "EmpID":empId,
    #     "FullName":fullName,
    #     "EmailID":emailID,
    # }
    # pd.DataFrame(data).to_sql("emp_details",con=conn,if_exists="append")
    query = f'INSERT INTO emp_details VALUES ({empId},"{fullName}","{emailID}")'
    # print(query)
    cursor.execute(query)
    conn.commit()
    conn.close()

def fetchDataFromDB(empID):
    conn = sqlite3.connect(config.SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"select * from emp_details where EmpID={empID}")
    details = cursor.fetchall()
    conn.close()
    return details

def save_frames_from_video(videoPath,frames_path):
    video_capture = cv2.VideoCapture(videoPath)
    cnt=0
    while True:
        cnt+=1
        ret, frame = video_capture.read()
        if ret:
            cv2.imwrite(os.path.join(frames_path,f"{cnt}.jpg"),frame)
        else:
            break


