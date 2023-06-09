# -*- coding: utf-8 -*-
"""Farid Wakim Thesis Code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1K19Tufu1HkeCzyBcuV4-AX6FPmAUiZgd

## This code represents the work of Farid Wakim(myself) to study emotion detection algorithms in order to improve it and allow it to become multilingual. It was done in order to achieve the Masters Degree in LIU.
"""

# Show: Avengers
# installing the needed external libraries and utilities
!apt install ffmpeg
!pip install soundfile
!pip3 install noisereduce
!pip install ffmpeg-python
!pip install imageio-ffmpeg
!pip install pydub
!pip install SpeechRecognition
!pip3 install librosa
!pip3 install tensorflow
!pip3 install pydub

# importing the os library helps us run os subproccesses
# importing the glob library to allow us to get the list of files in a directory
from glob import glob
# for sending commands from python to the command line
import subprocess
import os
# pandas and seaborn are data analysis and manipulation libraries
import pandas as pd
import seaborn as sns
# numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, 
# along with a large collection of high-level mathematical functions to operate on these arrays.
import numpy as np
# for plotting graphs
import matplotlib.pyplot as plt
# imageio is a Python library for video and audio manipulation
from pydub import AudioSegment
from pydub.silence import split_on_silence
# librosa is a python package for music and audio analysis.
import librosa as lb
import librosa.display
# noisereduce is a python library that allows us to easily remove the noise and background music
import noisereduce as nr

from scipy.io import wavfile

from pathlib import Path

import speech_recognition as sr



import pandas as pd
import numpy as np

import sys


!cp /content/drive/MyDrive/Masters/Audio_processing.py .
!cp /content/drive/MyDrive/Masters/config.py .
# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from tensorflow.keras import layers


import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
# from Read_Data import read_data
import Audio_processing
from Audio_processing import get_features
import config as conf

# we connected google colab to our google drive and set the main path
path = '/content/drive/MyDrive/Colab-Notebooks/data/'

!rm /content/drive/MyDrive/Colab-Notebooks/data/*-chunk*

!rm /content/drive/MyDrive/Colab-Notebooks/data/*reduced*

# we extract the audio using ffmpeg and a custom bash script 
!chmod 755 /content/drive/MyDrive/Colab-Notebooks/data/batch-convert.sh
!sh /content/drive/MyDrive/Colab-Notebooks/data/batch-convert.sh

# attempting to use noisereduce to decrease background noise
wavPath = path + '*.wav'
audioFiles = glob(wavPath)
for af in audioFiles:
  afn = af+"_reduced_noise.wav"
  if not os.path.exists(afn):
    # load data
    rate, data = wavfile.read(af)
    # perform noise reduction
    reduced_noise = nr.reduce_noise(y=data, sr=rate)
    wavfile.write(afn, rate, reduced_noise)
  audioFiles.remove(af)

# export on silence
wavPath = path + '*reduced_noise.wav'
audioFiles = glob(wavPath)
for af in audioFiles:
  sound = AudioSegment.from_wav(af)
  audio_chunks = split_on_silence(sound, min_silence_len=300, silence_thresh=-35 )
  for i, chunk in enumerate(audio_chunks):
    output_file = af[:-len(".wav")]+"-chunk{0}.wav".format(i)
    print("Exporting file", output_file)
    chunk.export(output_file, format="wav")

# plotting the difference between noisy file and cleaned up file
wavPath = path + '*reduced_noise*'
audioFiles = glob(wavPath)
for af in audioFiles:
  y, sr = lb.load(af)
  lb.feature.melspectrogram(y=y, sr=sr)
  D = np.abs(lb.stft(y))**2
  S = lb.feature.melspectrogram(S=D)
  plt.figure(figsize=(10, 4))
  lb.display.specshow(lb.power_to_db(S,ref=np.max), y_axis='mel', fmax=8000,x_axis='time')
  plt.colorbar(format='%+2.0f dB')
  plt.title('Mel spectrogram for '+af)
  plt.tight_layout()

Ravdess= "/content/drive/MyDrive/Masters/Ravdess/audio_speech_actors_01-24/"
ravdess_directory_list = os.listdir(Ravdess)

file_emotion = []
file_path = []
for dir in ravdess_directory_list:
    # as their are 20 different actors in our previous directory we need to extract files for each actor.
    actor = os.listdir(Ravdess + dir)
    for file in actor:
        part = file.split('.')[0]
        part = part.split('-')
        # third part in each file represents the emotion associated to that file.
        file_emotion.append(int(part[2]))
        file_path.append(Ravdess + dir + '/' + file)
        
# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Ravdess_df = pd.concat([emotion_df, path_df], axis=1)

# changing integers to actual emotions.
Ravdess_df.Emotions.replace({1:'neutral', 2:'neutral', 3:'happy', 4:'sad', 5:'angry', 6:'fear', 7:'disgust', 8:'surprise'}, inplace=True)



X, Y = [], []
for path, emotion in zip(Ravdess_df.Path, Ravdess_df.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv('features_emotion.csv', index=False)

Features=Features.fillna(0)

X = Features.drop(labels = 'labels', axis =1)
Y = Features['labels']

# As this is a multiclass classification problem onehotencoding our Y.
lb=LabelEncoder()
Y=np_utils.to_categorical(lb.fit_transform(Y))

# splitting data
X_train,X_test,y_train,y_test=train_test_split(X,Y,random_state=42,test_size=0.2,shuffle=True)

X_train,X_val,y_train,y_val=train_test_split(X_train,y_train,random_state=42,test_size=0.1,shuffle=True)

# scaling our data with sklearn's Standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# making our data compatible to model.
X_train=np.expand_dims(X_train,axis=2)
X_val=np.expand_dims(X_val,axis=2)
X_test=np.expand_dims(X_test,axis=2)

model=tf.keras.Sequential([
    layers.Conv1D(conf.NN_parameters['neurons'][6],kernel_size=5, strides=1,padding='same', activation='relu',input_shape=(X_train.shape[1],1)),
    layers.BatchNormalization(),
    layers.MaxPool1D(pool_size=5,strides=2,padding='same'),
    layers.Conv1D(conf.NN_parameters['neurons'][5],kernel_size=5,strides=1,padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool1D(pool_size=5,strides=2,padding='same'),
    layers.Conv1D(conf.NN_parameters['neurons'][4],kernel_size=3,strides=1,padding='same',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPool1D(pool_size=3,strides=2,padding='same'),
    layers.Flatten(),
    layers.Dense(conf.NN_parameters['neurons'][6],activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(conf.NN_parameters['neurons'][0]-1,activation='softmax')
])
model.compile(optimizer=conf.NN_parameters['optimizers'][0],loss=conf.NN_parameters['loss_functions'][2],metrics=conf.NN_parameters['metrics'][0])

early_stop=EarlyStopping(monitor=conf.NN_parameters['monitors'][0],mode='auto',patience=conf.NN_parameters['patience'][4],restore_best_weights=True)
lr_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=conf.NN_parameters['patience'][2],verbose=1,factor=conf.NN_parameters['learning_rate'][4],min_lr=conf.NN_parameters['learning_rate'][4])

checkpoint = ModelCheckpoint(conf.Output_models_path['Emotion_model']+"Emotion.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
EPOCH=5
BATCH_SIZE=64

history=model.fit(X_train,y_train, epochs=EPOCH,validation_data=(X_val, y_val), callbacks=[checkpoint,early_stop,lr_reduction], batch_size = BATCH_SIZE)
model.save("/Trained_models/Best_Model_Emotion_Recog.h5")