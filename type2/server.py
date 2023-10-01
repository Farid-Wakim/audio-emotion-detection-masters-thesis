from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path
import os, glob
from glob import glob
from io import StringIO

import subprocess, os
import pandas as pd
# import seaborn as sns
import tensorflow as tf
import librosa as lb

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import GlobalMaxPooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model

from scipy.io import wavfile
from pydub import AudioSegment

import noisereduce as nr


app = Flask(__name__)

model=""
train=""
test = ""



def transformAudio(audioFileName):
    audio = AudioSegment.from_file(audioFileName)
    audio.export("type2/files/test_input.wav", format="wav")
    app.logger.info('noise being reduced')

    # perform noise reduction
    time.sleep(2.4)
    rate, data = wavfile.read("type2\\files\\test_input.wav")
    orig_shape = data.shape
    data = np.reshape(data, (2, -1))
    reduced_noise = nr.reduce_noise(y=data,sr=rate,stationary=True)
    wavfile.write("type2\\files\\test_input_noise_reduced.wav", rate, reduced_noise.reshape(orig_shape))



def generateSpectos():
    wavPath = 'type2\\files\\*.wav'
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
        plt.savefig('static\\spc'+af+'.png')





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
                model = f            
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
                    video.export("type2/files/test_input.wav", format="wav")
                    transformAudio(f.filename)
                    generateSpectos()
                    return render_template('processVideo.html')
                
                elif(extension == '.mp3' or extension == '.wav' or extension == '.wma'):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'type2/files/test'+ extension            
                    f.save(f.filename)
                    transformAudio(f.filename)
                    generateSpectos()
                    return render_template('processAudio.html')
                
                elif(extension == '.txt' or extension == '.json' ):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'type2/files/test'+ extension            
                    f.save(f.filename)
                    return render_template('processText.html')
  
                else:
                    app.logger.info('invalid extension  --> '+extension)
                    return render_template('uploadModel.html', error="Incorrect Format")
        else:
            return render_template('uploadModel.html', error="Unexpected Error")


@app.route('/processVideo')
def processVideo():
    return send_file("files\\test_input_noise_reduced.wav",as_attachment=True)


@app.route('/processAudio')
def processAudio():    
    return send_file("files\\test_input_noise_reduced.wav",as_attachment=True)

#process text
@app.route('/processText', methods=['POST'])
def processText():
    return render_template('processText.html')


#training In Progress
@app.route('/trainingInProgress', methods=['POST'])
def trainingInProgress():
    return render_template('trainingInProgress.html')


#process video
@app.route('/showResult')
def showResult():
    savedModel=load_model('type2\\files\\model.h5')
    origStdout = sys.stdout
    outputBuf = StringIO()
    sys.stdout = outputBuf
    savedModel.summary()
    sys.stdout = origStdout
    Get_model_description = outputBuf.getvalue()
    savedModel.predict("thingamabob")
    print(Get_model_description)
    return render_template('showResult.html', summary=' '+summary)




if __name__ == '__main__':
    app.run(debug=True)

