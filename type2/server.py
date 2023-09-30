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
from scipy.io import wavfile
from pydub import AudioSegment

import noisereduce as nr


app = Flask(__name__)

model=""
train=""
test = ""



def transformAudio(audioFileName):
    audio = AudioSegment.from_file(audioFileName)
    audio.export("type2/files/test.wav", format="wav")
    app.logger.info('noise being reduced')

    # perform noise reduction
    time.sleep(2.4)
    rate, data = wavfile.read("type2\\files\\test.wav")
    orig_shape = data.shape
    data = np.reshape(data, (2, -1))
    reduced_noise = nr.reduce_noise(y=data,sr=rate,stationary=True)
    wavfile.write("type2\\files\\test_nr.wav", rate, reduced_noise.reshape(orig_shape))


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
                    video.export("type2/files/test.wav", format="wav")
                    transformAudio(f.filename)
                    return render_template('processVideo.html')
                
                elif(extension == '.mp3' or extension == '.wav' or extension == '.wma'):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'type2/files/test'+ extension            
                    f.save(f.filename)
                    transformAudio(f.filename)

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
    return send_file("files\\test_nr.wav",as_attachment=True)


@app.route('/processAudio')
def processAudio():    
    return send_file("files\\test_nr.wav",as_attachment=True)





# #process audio
# @app.route('/processAudio', methods=['POST'])
# def processAudio():
#     return render_template('processAudio.html')


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

