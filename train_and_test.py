from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import tensorflow as tf

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



def transformAudio(audioFileName):
    audio = AudioSegment.from_file(audioFileName)
    audio.export("files/test_input.wav", format="wav")
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
    for filename in glob("files/*"):
        os.remove(filename) 
        print("All files deleted")
    return render_template('index.html')

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
                f.filename = 'files/model'+ extension            
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
                    f.filename = 'files/test'+ extension            
                    f.save(f.filename)
                    video = AudioSegment.from_file(f.filename)
                    video.export("files/test_input.wav", format="wav")
                    transformAudio(f.filename)
                    generateSpectos()
                    return render_template('processVideo.html')
                
                elif(extension == '.mp3' or extension == '.wav' or extension == '.wma'):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'files/test'+ extension            
                    f.save(f.filename)
                    transformAudio(f.filename)
                    generateSpectos()
                    return render_template('processAudio.html')
                
                elif(extension == '.txt' or extension == '.json' ):
                    app.logger.info('extension valid -->' +extension)
                    f.filename = 'files/test'+ extension            
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


    
    print("Num GPUs Available: ", len(list_physical_devices('GPU')))

    RAVDESS_Dataset_class = RAVDESS_Dataset()
    X, Y = RAVDESS_Dataset_class.create_dataset()

    # As this is a multiclass classification problem onehotencoding our Y.
    encoder = OneHotEncoder()
    Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, train_size=0.70, random_state=0, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, train_size=0.5, random_state=0, shuffle=True)
    print("X train:" ,np.asarray(x_train).shape)
    print("Y train:" ,y_train.shape)
    print("X test:" ,x_test.shape)
    print("Y test:" ,y_test.shape)
    print("X val:" ,x_val.shape)
    print("X val:" ,y_val.shape)

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape

    # making our data compatible to model.
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    x_val = np.expand_dims(x_val, axis=2)
    x_train.shape, y_train.shape, x_test.shape, y_test.shape

    model = get_model(X.shape[1])

    epochs = 100
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.8, verbose=1, patience=15, min_lr=0.000001) #

    history=model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_data=(x_val, y_val), callbacks=[rlrp])

    model.save('files\\model.h5')

    #-------------------- TEST --------------------#
    model = load_model('model.h5')

    print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")

    epochs = [i for i in range(epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    test_acc = history.history['val_accuracy']
    test_loss = history.history['val_loss']

    fig.set_size_inches(20,8)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , test_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , test_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.savefig('train_loss_accuracy.png')

    
    pred_test = model.predict(x_test)
    y_pred = encoder.inverse_transform(pred_test)

    y_test = encoder.inverse_transform(y_test)

    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = y_pred.flatten()
    df['Actual Labels'] = y_test.flatten()

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in encoder.categories_] , columns = [i for i in encoder.categories_])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('static\\confusion_matrix.png')
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

