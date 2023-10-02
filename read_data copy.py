from scipy.io import wavfile
import noisereduce as nr
import numpy as np

# load data
rate, data = wavfile.read("type2\\files\\test.wav")
orig_shape = data.shape
data = np.reshape(data, (2, -1))

# perform noise reduction
# optimized for speech
reduced_noise = nr.reduce_noise(
    y=data,
    sr=rate,
    stationary=True
)

wavfile.write("type2\\files\\test_nr.wav", rate, reduced_noise.reshape(orig_shape))
