Audio_Processing_Parameters =  {
    "rate" : 0.8,
    "sampling_rate" : 44100,
    "pitch_factor" : 0.7,
    "type":['noise','stretch','shift','pitch'],
    "frame_length":2048,
    "hop_length":512,
    "duration":5,
    "offset":0.5
}

CSV_files_saved =  {
    "data_path_Emotions": "EmotionModelGenerator/Output/"
}

Dataset_Paths = {
    "Ravdess": "/content/drive/MyDrive/Masters/Ravdess/audio_speech_actors_01-24/"
}
Output_models_path = {
    "Emotion_model":"Output/"
}

NN_parameters = {
    "neurons":[8,16,32,64,128,256,512,1024],
    "loss_functions":["mean_squared_error","binary_crossentropy","categorical_crossentropy"],
    "optimizers": ["adam","sgd"],
    "metrics":["accuracy","mse","mae"],
    "monitors":["val_accuracy","val_mse","val_mae"],
    "patience":[1,2,3,4,5,6,7,10],
    "learning_rate":[0.00001,0.0005,0.001,0.05, 0.1],
    "factor":[0.1,0.2,0.3,0.4,0.5]

}
sleeping_time = 10