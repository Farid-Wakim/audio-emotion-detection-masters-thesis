from distutils.log import debug
from fileinput import filename
from flask import *
import time
import os.path
import os, glob
from glob import glob

import subprocess, os
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.silence import split_on_silence
import librosa as lb
import librosa.display
import noisereduce as nr
import keras
import tensorflow as tf


from scipy.io import wavfile

from pathlib import Path

import sys


from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split





app = Flask(__name__)


@app.route('/')
def main():
    for filename in glob.glob("files/model*"):
        os.remove(filename) 
    for filename in glob.glob("files/data*"):
        os.remove(filename) 
    return render_template('index.html')


@app.route('/modelUploaded', methods=['POST'])
def modelUploaded():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename.__len__() > 0): 
            extension = os.path.splitext(f.filename)[1]
            # timeEq = str(time.time())
            f.filename = 'files/data'+ extension            
            f.save(f.filename)
            return render_template('dataUploaded.html')
        else:
            return render_template('index.html')


@app.route('/dataUploaded', methods=['POST'])
def dataUploaded():
    if request.method == 'POST':
        f = request.files['file']
        if(f.filename.__len__() > 0):
            extension = os.path.splitext(f.filename)[1]
            # timeEq = str(time.time())
            f.filename = 'files/data'+ extension            
            f.save(f.filename)
            return render_template('testResults.html')
        else:
            return render_template('index.html')




@app.route('/prepareModel')
def prepareModel():
    return render_template('prepareModel.html')

@app.route('/prepareConfig' , methods=['POST'])
def configUploaded():
    if request.method == 'POST':
        f = request.form.get('config')
        print(f)
        
    return render_template('configUpdated.html')


if __name__ == '__main__':
    app.run(debug=True)

