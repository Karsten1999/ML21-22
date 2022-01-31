import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
sys.path.append(os.path.abspath('../general'))
import Preprocessing
import Postprocessing
import Convert_to_MIDI
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

voice1 = np.loadtxt("../data/F.txt").T[0][1500::]
voice2 = np.loadtxt("../data/F.txt").T[1][1500::]
voice3 = np.loadtxt("../data/F.txt").T[2][1500::]
voice4 = np.loadtxt("../data/F.txt").T[3][1500::]

vector1 = Preprocessing.Transform_into_vector(voice1)
vector2 = Preprocessing.Transform_into_vector(voice2)
vector3 = Preprocessing.Transform_into_vector(voice3)
vector4 = Preprocessing.Transform_into_vector(voice4)

predicted_voice1 = []
predicted_voice2 = []
predicted_voice3 = []
predicted_voice4 = []

# save the lengths of the individual vectors into array
vector_lengths = [len(vector1[0]), len(vector2[0]), len(vector3[0]), len(vector4[0])]
min_note = [min(voice1[voice1>0]), min(voice2[voice2>0]), min(voice3[voice3>0]), min(voice4[voice4>0])]


# paste into big vector
vector = np.hstack([vector1, vector2, vector3, vector4])
# paste voice into large vector of voices, alternating
voice = [item for sublist in zip(voice1, voice2, voice3, voice4) for item in sublist]

score = []
difference = []

prediction_length = 256
window_size = 250*4
# window = range(1,250*4,1)

X, y = Preprocessing.Split_rolling_window(voice, vector, window_size=window_size, train=False)

scaler = StandardScaler()
scaler.fit(X)
reg = LinearRegression().fit(scaler.transform(X), y)

for i in range(prediction_length):
    length = len(voice)
    X = [voice[length-window_size::]]
    y_probs_pred = reg.predict(scaler.transform(X))
    
    m = 0
    #separate y_pred into individual vectors and pick highest value
    for j in range(4):
        n = vector_lengths[j]
        y_pred_j = Postprocessing.prediction_vector_pick_highest(y_probs_pred[:, m:m+n], zero_bias=0.1)[0]
        if m == 0:
            y_pred = y_pred_j
        else:
            y_pred = np.hstack([y_pred, y_pred_j])                
        m += n
        
        # Converting vector to note number
        note = Postprocessing.vector_to_note(min_note[j], y_pred_j)
        if j == 0:
            predicted_voice1.append(note)
            voice1 = np.append(voice1, note)
        if j == 1:
            predicted_voice2.append(note)
            voice2 = np.append(voice2, note)
        if j == 2:
            predicted_voice3.append(note)
            voice3 = np.append(voice3, note)
        if j == 3:
            predicted_voice4.append(note)
            voice4 = np.append(voice4, note)
    
    voice = [item for sublist in zip(voice1, voice2, voice3, voice4) for item in sublist]
    vector = np.append(vector, y_pred)

four_voice_vector = np.vstack([voice1, voice2, voice3, voice4])
four_voice_predicted_vector = np.vstack([predicted_voice1, predicted_voice2, 
                                       predicted_voice3, predicted_voice4])

np.savetxt("output_all_voices_complete.txt", four_voice_vector.T.astype(int), fmt='%s')
np.savetxt("output_all_voices_new.txt", four_voice_predicted_vector.T, fmt='%s')

Convert_to_MIDI.convert_to_midi("output_all_voices_complete.txt", "output_all_voices_complete.mid")
Convert_to_MIDI.convert_to_midi("output_all_voices_new.txt", "output_all_voices_new.mid")