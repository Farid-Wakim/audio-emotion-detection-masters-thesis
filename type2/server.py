from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path
import os, glob
from glob import glob

import subprocess, os
import pandas as pd
# import seaborn as sns
import numpy as np
# import tensorflow as tf

# import matplotlib.pyplot as plt
from pydub import AudioSegment
# from pydub.silence import split_on_silence
# import librosa as lb
# import librosa.display
# import noisereduce as nr
# import keras
# import tensorflow as tf


# from scipy.io import wavfile

# from pathlib import Path

# import sys


# from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.model_selection import train_test_split





app = Flask(__name__)

model=""
train=""
test = ""


@app.route('/')
def main():
    # resets all files
    for filename in glob("type2/files/*"):
        os.remove(filename) 
        print("All files deleted")
    return render_template('index.html')
# https://www.youtube.com/playlist?list=PLGnR4DOsNUPKDTOPPLyHFtj2owHQpUXRC

#prepare or upload model
@app.route('/chooseModel')
def chooseModel():
    return render_template('chooseModel.html')



#prepare model from configuration by entering the configuration
@app.route('/prepareModel')
def prepareModel():
    return render_template('prepareModel.html')


#uploaded premade model, now display it
@app.route('/uploadModel', methods=['POST'])
def uploadModel():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename.__len__() > 0): 
            extension = os.path.splitext(f.filename)[1]
            if(extension == '.h5' or extension == '.yml' or extension == '.json'):
                app.logger.info('extension valid -->' +extension)
                f.filename = 'type2/files/model'+ extension            
                f.save(f.filename)
                return render_template('uploadModel.html')
            else:
                app.logger.info('invalid extension  --> '+extension)
                return render_template('chooseModel.html', error="Incorrect Format")
        else:
            return render_template('chooseModel.html', error="Unexpected Error")



#upload Training
@app.route('/uploadTraining', methods=['POST'])
def uploadTraining():
    # training
    return render_template('uploadTraining.html')


#upload Test video or audio
@app.route('/uploadTest', methods=['POST'])
def uploadTest():
        if request.method == 'POST':
            f = request.files['file']
            if(f.filename.__len__() > 0): 
                extension = os.path.splitext(f.filename)[1]
                if(extension == '.mp4' or extension == '.mov' or extension == '.avi'):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'type2/files/test'+ extension            
                    f.save(f.filename)
                    video = AudioSegment.from_file(f.filename)
                    video.export("type2/files/test.mp3", format="mp3")

                    return render_template('processVideo.html')
                elif(extension == '.mp3' or extension == '.wav' or extension == '.wma'):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'type2/files/test'+ extension            
                    f.save(f.filename)
                    return render_template('processAudio.html')
                
                else:
                    app.logger.info('invalid extension  --> '+extension)
                    return render_template('uploadModel.html', error="Incorrect Format")
        else:
            return render_template('uploadModel.html', error="Unexpected Error")


@app.route('/processVideo')
def processVideo():
    return send_file("files\\test.mp3",as_attachment=True)

    




#process video
# @app.route('/processVideo', methods=['POST'])
# def processVideo():
#     return render_template('processVideo.html')





#process audio
@app.route('/processAudio', methods=['POST'])
def processAudio():
    return render_template('processAudio.html')


#process text
@app.route('/processText', methods=['POST'])
def processText():
    return render_template('processText.html')


#training In Progress
@app.route('/trainingInProgress', methods=['POST'])
def trainingInProgress():
    return render_template('trainingInProgress.html')


#process video
@app.route('/showResult', methods=['POST'])
def showResult():
    return render_template('showResult.html')



if __name__ == '__main__':
    app.run(debug=True)

