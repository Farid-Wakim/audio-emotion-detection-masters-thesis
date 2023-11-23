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

from scipy.io import wavfile
from pydub import AudioSegment

import noisereduce as nr

app = Flask(__name__)

class_labels=['neutral','calm','happy','sad','angry','fear','disgust','surprise']
uploaded_file=""
#transform video file to audio.wav
def transformAudio(audioFileName):
    audio = AudioSegment.from_file(audioFileName)
    audio.export("test/processed/test_input.wav", format="wav")
    app.logger.info('noise being reduced')

    # perform noise reduction
    time.sleep(2.4)
    rate, data = wavfile.read("test/processed/test_input.wav")
    orig_shape = data.shape
    data = np.reshape(data, (2, -1))
    reduced_noise = nr.reduce_noise(y=data,sr=rate,stationary=True)
    wavfile.write("test/processed/test_input_noise_reduced.wav", rate, reduced_noise.reshape(orig_shape))



def generateSpectos():
    wavPath = 'files/*.wav'
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
        plt.savefig('static/spc'+af+'.png')




@app.route('/')
def main():
    # resets all files
    # for filename in glob("files/*"):
    #     os.remove(filename) 
    #     print("All files deleted")
    return render_template('index.html')

#prepare or upload model
@app.route('/chooseModel')
def chooseModel():
    return render_template('chooseModel.html')



#prepare model from configuration by entering the configuration
# @app.route('/prepareModel')
# def prepareModel():
#     return render_template('prepareModel.html')


#uploaded premade model, now display it
@app.route('/uploadModel', methods=['POST'])
def uploadModel():
    if request.method == 'POST':
        f = request.files['file']
        print("uploaded file ",uploaded_file)
        if(f.filename.__len__() > 0): 
            extension = os.path.splitext(f.filename)[1]
            if(extension == '.h5' or extension == '.yml' or extension == '.json' ):
                app.logger.info('extension valid -->' +extension)
                print(extension)
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
# @app.route('/uploadTest', methods=['POST'])
# def uploadTest():
#         if request.method == 'POST':
#             f = request.files['file']
#             os.makedirs("test/processed",exist_ok=True)
#             if(f.filename.__len__() > 0): 
                
#                 extension = os.path.splitext(f.filename)[1]
#                 fname=os.path.splitext(f.filename)[0]
          
#                 if(extension == '.mp4' or extension == '.mov' or extension == '.avi' or extension =='.flv'):
#                     app.logger.info('extension valid -->' +extension)
#                     f.filename = 'test/processed/'+fname+ extension            
#                     f.save(f.filename)
#                     video = AudioSegment.from_file(f.filename)
#                     video.export(f"test/processed/{fname}.wav", format="wav")
#                     transformAudio(f.filename)
#                     # generateSpectos()
#                     return render_template('processVideo.html')
                
#                 elif(extension == '.mp3' or extension == '.wav' or extension == '.wma'):
#                     app.logger.info('extension valid -->' +extension)
#                     f.filename = 'test/processed/'+fname+ extension            
#                     f.save(f.filename)
#                     # transformAudio(f.filename)
#                     generateSpectos()
#                     return render_template('processAudio.html')
                
#                 elif(extension == '.txt' or extension == '.json' ):
#                     app.logger.info('extension valid -->' +extension)
#                     f.filename = 'test/processed/'+fname+ extension            
#                     f.save(f.filename)
#                     return render_template('processText.html')
  
#                 else:
#                     app.logger.info('invalid extension  --> '+extension)
#                     return render_template('uploadModel.html', error="Incorrect Format")
#         else:
#             return render_template('uploadModel.html', error="Unexpected Error")



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
            f"{output_folder}/chunk{i}.wav",
            bitrate = "192k",
            format = "wav"
        )

@app.route('/uploadTest', methods=['POST'])
def uploadTest():
        if request.method == 'POST':
            f = request.files['file']
            os.makedirs("test/processed",exist_ok=True)
            if(f.filename.__len__() > 0): 
                
                extension = os.path.splitext(f.filename)[1]
                fname=os.path.splitext(f.filename)[0]
                
                if(extension == '.mp3' or extension == '.wav' or extension == '.wma' or extension == '.mp4' or extension == '.mov' or extension == '.avi' or extension =='.flv'):
                    app.logger.info('extension valid -->' +extension)
                    split_audio_on_silence(f"test/{f.filename}","test/processed")
                    return render_template('processVideo.html')
                
                else:
                    app.logger.info('invalid extension  --> '+extension)
                    return render_template('uploadModel.html', error="Incorrect Format")
        else:
            return render_template('uploadModel.html', error="Unexpected Error")


@app.route('/processVideo')
def processVideo():
    return send_file("test/processed/test_input_noise_reduced.wav",as_attachment=True)


@app.route('/processAudio')
def processAudio():    
    return send_file("test/processed/test_input_noise_reduced.wav",as_attachment=True)

#process text
@app.route('/processText', methods=['POST'])
def processText():
    return render_template('processText.html')


#training In Progress
# @app.route('/trainingInProgress', methods=['POST'])
@app.route('/prepareModel')
def trainingInProgress():

    print("Num GPUs Available: ", len(list_physical_devices('GPU')))
    RAVDESS_Dataset_class = RAVDESS_Dataset()
    X, Y = RAVDESS_Dataset_class.create_dataset()

    a=f"""  loaded X: {X}\n 
                   Y: {Y}\n 
            shape  X: {X.shape}\n
                   Y: {Y.shape}"""
    print(a)
    # As this is a multiclass classification problem onehotencoding our Y.
    # encoder = OneHotEncoder()
    # Y = encoder.fit_transform(np.array(Y).reshape(-1,1)).toarray()
    y=np.zeros(Y.shape[0])
    for i,sample in enumerate(Y):
        y[i]=class_labels.index(sample)
    print("y ==>",y)

    # splitting data
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.30, train_size=0.70, random_state=1, shuffle=True)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, train_size=0.5, random_state=1, shuffle=True)
    # print("X train", x_train.shape)
    # print("X train:" ,np.asarray(x_train).shape)
    # print("Y train:" ,y_train.shape)
    # print("X test:" ,x_test.shape)
    # print("Y test:" ,y_test.shape)
    # print("X val:" ,x_val.shape)
    # print("X val:" ,y_val.shape)

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

    epochs = 300
    rlrp = ReduceLROnPlateau(monitor='loss', factor=0.2, verbose=1, patience=5, min_lr=0.000001) #

    history=model.fit(x_train, y_train, batch_size=28, epochs=epochs, validation_data=(x_val, y_val), callbacks=[rlrp])

    #saving model weights and history
    model.save('files/model.h5')
    
    hist_df = pd.DataFrame(history.history)

    with open("files/history.json", "w") as outfile:
        hist_df.to_json(outfile)
    
    print("***Ploting***")
    epochs = [i for i in range(epochs)]
    fig , ax = plt.subplots(1,2)
    train_acc = history.history['accuracy']
    train_loss = history.history['loss']
    val_acc = history.history['val_accuracy']
    val_loss = history.history['val_loss']

    
    fig.set_size_inches(20,8)
    ax[0].plot(epochs , train_loss , label = 'Training Loss')
    ax[0].plot(epochs , val_loss , label = 'Testing Loss')
    ax[0].set_title('Training & Testing Loss')
    ax[0].legend()
    ax[0].set_xlabel("Epochs")

    ax[1].plot(epochs , train_acc , label = 'Training Accuracy')
    ax[1].plot(epochs , val_acc , label = 'Testing Accuracy')
    ax[1].set_title('Training & Testing Accuracy')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    plt.savefig('train_loss_accuracy.png')


    print('#-------------------- TEST --------------------#')

    print("***Testing***")
    
    model = load_model('files/model.h5')
    print("Accuracy of our model on test data : " , model.evaluate(x_test,y_test)[1]*100 , "%")
  

    pred_test = model.predict(x_test)
    predicted_class_indices = np.argmax(pred_test, axis=1)

   
    y_pred = [class_labels[i] for i in predicted_class_indices]
    
    y_test = np.array(y_test).astype(int)
    y_test= [class_labels[i] for i in y_test]

    # y_pred = encoder.inverse_transform(pred_test)
    # y_test = encoder.inverse_transform(y_test)

    
    df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df['Predicted Labels'] = np.array(y_pred).flatten()
    df['Actual Labels'] = np.array(y_test).flatten()
    print("df: ",df)
    with open("files/test_prediction.csv", "w") as outfile:
        df.to_csv(outfile)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize = (12, 10))
    cm = pd.DataFrame(cm , index = [i for i in class_labels] , columns = [i for i in class_labels])
    sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
    plt.title('Confusion Matrix', size=20)
    plt.xlabel('Predicted Labels', size=14)
    plt.ylabel('Actual Labels', size=14)
    plt.savefig('static/confusion_matrix.png')
    return render_template('trainingInProgress.html')



#process video
@app.route('/showResult')
def showResult():
    dataset=RAVDESS_Dataset()
    
    model=load_model('files/model.h5')
    origStdout = sys.stdout
    outputBuf = StringIO()
    sys.stdout = outputBuf
    model.summary()
    sys.stdout = origStdout
    Get_model_description = outputBuf.getvalue()
    print("uploaded_file:",uploaded_file)
    X,Y=dataset.load_test("test/processed")

    print("GT--------->",Y)
    a=f"""  loaded X: {X}\n 
                   Y: {Y}\n 
            shape  X: {X.shape}\n
                   Y: {Y.shape}"""
    print(a)
          

    # scaling our data with sklearn's Standard scaler
    scaler = StandardScaler()
    x = scaler.fit_transform(X)
   

    # making our data compatible to model.
    x = np.expand_dims(x, axis=2)

    
    pred=model.predict(x)

    # Get the predicted class indices
    predicted_class_indices = np.argmax(pred, axis=1)
    #  Map predicted indices to class labels
    y_pred = [class_labels[i] for i in predicted_class_indices]

    print(y_pred)

    # df = pd.DataFrame(columns=['Predicted Labels', 'Actual Labels'])
    df = pd.DataFrame(columns=['Predicted Labels'])
    predicted_values=df

    df['Predicted Labels'] = np.array(y_pred).flatten()
    

    # df['Actual Labels'] = Y.flatten()
    
    with open("prediction.csv", "w") as outfile:
       df.to_csv(outfile)
    
    output=' '.join(y_pred[0])
   
    print("df",df)
    # for i,value in enumerate(predicted_values.items()):

    print(Get_model_description)

    return render_template('showResult.html',predicted_values=predicted_values)




if __name__ == '__main__':
    app.run(debug=True)

