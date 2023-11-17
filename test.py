from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import tensorflow as tf
import json
from model import get_model
from dataset import RAVDESS_Dataset
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.config import list_physical_devices
from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path
import os, glob
from glob import glob
from io import StringIO
import sys
import subprocess, os
import pandas as pd
# import seaborn as sns
import tensorflow as tf
import librosa as lb

import pydub
from pydub import AudioSegment
from pydub.silence import split_on_silence

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model



def inverse_sparse_one_hot_encoding(data, encoding_dict):
    num_samples = len(data)
     
    # Initialize an array with zeros
    decoded_array = []
    for i, sample in enumerate(data):
        for key,value in encoding_dict.items():
            if sample==value:
                print(key)
                decoded_array.append(key)
                
    return decoded_array

def match_target_amplitude(aChunk, target_dBFS):
    ''' Normalize given audio chunk '''
    change_in_dBFS = target_dBFS - aChunk.dBFS
    return aChunk.apply_gain(change_in_dBFS)

def split_audio_on_silence(input_file, output_folder, silence_duration=1500, silence_threshold=-50):
    # Load your audio.
    song = AudioSegment.from_file(input_file)

    # Split track where the silence is 2 seconds or more and get chunks using 
    # the imported function.
    chunks = split_on_silence (
        # Use the loaded audio.
        song, 
        # Specify that a silent chunk must be at least 2 seconds or 2000 ms long.
        min_silence_len = silence_duration,
        # Consider a chunk silent if it's quieter than -16 dBFS.
        # (You may want to adjust this parameter.)
        silence_thresh = silence_threshold
    )

    # Process each chunk with your parameters
    for i, chunk in enumerate(chunks):
        # Create a silence chunk that's 0.5 seconds (or 500 ms) long for padding.
        silence_chunk = AudioSegment.silent(duration=500)

        # Add the padding chunk to beginning and end of the entire chunk.
        audio_chunk = silence_chunk + chunk + silence_chunk

        # Normalize the entire chunk.
        normalized_chunk = match_target_amplitude(audio_chunk, -20.0)

        # Export the audio chunk with new bitrate.
        print("Exporting chunk{0}.wav.".format(i))
        normalized_chunk.export(
            "test/processed/chunk{0}.wav".format(i),
            bitrate = "192k",
            format = "wav"
        )



if __name__=="__main__":


    split_audio_on_silence(input_file="test/The Pursuit Of Happiness.flv",output_folder="test")

    # dataset=RAVDESS_Dataset()
    # model=load_model('files/model.h5')
    # X,Y=dataset.load_test("test")

    # print("GT--------->",Y)
    # a=f"""  loaded X: {X}\n 
    #                Y: {Y}\n 
    #         shape  X: {X.shape}\n
    #                Y: {Y.shape}"""
    # print(a)
          
    # class_labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise']

    # # scaling our data with sklearn's Standard scaler
    # scaler = StandardScaler()
    # x = scaler.fit_transform(X)
   

    # # making our data compatible to model.
    # x = np.expand_dims(x, axis=2)

    
    # pred=model.predict(x)

    # # Get the predicted class indices
    # predicted_class_indices = np.argmax(pred, axis=1)
    # #  Map predicted indices to class labels
    # y_pred = [class_labels[i] for i in predicted_class_indices]

    # print(y_pred)

    # df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    # df['Predicted Labels'] = np.array(y_pred).flatten()
    # df['Actual Labels'] = Y.flatten()
    
    # with open("t.csv", "w") as outfile:
    #    df.to_csv(outfile)

