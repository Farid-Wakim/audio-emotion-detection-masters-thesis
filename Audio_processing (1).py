import librosa
import numpy as np
import config as conf

# DATA AUGMENTATION
def data_augmentation(type, data, rate=conf.Audio_Processing_Parameters['rate'], sampling_rate = conf.Audio_Processing_Parameters['sampling_rate'], pitch_factor=conf.Audio_Processing_Parameters['pitch_factor']):
    if (type == conf.Audio_Processing_Parameters['type'][0]):
        noise_amp = 0.035*np.random.uniform()*np.amax(data)
        data = data + noise_amp*np.random.normal(size=data.shape[0])
        return data
    elif (type == conf.Audio_Processing_Parameters['type'][1]):
        return librosa.effects.time_stretch(y=data, rate = rate)
    elif (type == conf.Audio_Processing_Parameters['type'][2]):
        shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
        return np.roll(data, shift_range)
    elif (type == conf.Audio_Processing_Parameters['type'][3]):
        return librosa.effects.pitch_shift(y = data,sr = sampling_rate, n_steps = pitch_factor)
    else:
        print('wrong input')

# FEATURES EXTRACTION
def zcr(data,frame_length,hop_length):
    zcr=librosa.feature.zero_crossing_rate(y = data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(zcr)

def rmse(data,frame_length=conf.Audio_Processing_Parameters['frame_length'],hop_length=conf.Audio_Processing_Parameters['hop_length']):
    rmse=librosa.feature.rms(y = data,frame_length=frame_length,hop_length=hop_length)
    return np.squeeze(rmse)

def mfcc(data,sr,flatten:bool=True):
    mfcc=librosa.feature.mfcc(y = data,sr=sr)
    return np.squeeze(mfcc.T)if not flatten else np.ravel(mfcc.T)

def extract_features(data,sr,frame_length=conf.Audio_Processing_Parameters['frame_length'],hop_length=conf.Audio_Processing_Parameters['hop_length']):
    result=np.array([])
    
    result=np.hstack((result,
                      zcr(data,frame_length,hop_length),
                      rmse(data,frame_length,hop_length),
                      mfcc(data,sr)
                     ))
    return result

def get_features(path,duration=conf.Audio_Processing_Parameters['duration'], offset=conf.Audio_Processing_Parameters['offset']):
    data,sr=librosa.load(path,duration=duration,offset=offset)
    aud=extract_features(data,sr)
    audio=np.array(aud)
    
    noised_audio=data_augmentation(type = 'noise',data=data)
    aud2=extract_features(noised_audio,sr)
    audio=np.vstack((audio,aud2))
    
    pitched_audio=data_augmentation(type = 'pitch',data = data)
    aud3=extract_features(pitched_audio,sr)
    audio=np.vstack((audio,aud3))
    
    pitched_audio1=data_augmentation(type = 'pitch',data = data)
    pitched_noised_audio=data_augmentation(type = 'noise',data = pitched_audio1)
    aud4=extract_features(pitched_noised_audio,sr)
    audio=np.vstack((audio,aud4))
    
    return audio