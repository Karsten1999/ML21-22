import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
sys.path.append(os.path.abspath('../general'))
import Preprocessing
import Postprocessing
import Convert_to_MIDI
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

def create_track(tracknumber: int, window_size = 500):
    # 500,600, 650, 680 was good, 650 best probably for window size
    # data
    voice1 = np.loadtxt("../data/F.txt").T[tracknumber-1][3095::]
    vector = Preprocessing.Transform_into_vector(voice1)


    # minimum note used for converting probability vector back into note
    min_note = min(voice1[voice1>0])

    predicted_voice = []

    # Parameter to set length of prediction
    prediction_length = 256

    X, y = Preprocessing.Split_rolling_window(voice1, vector, window_size=window_size, train=False)

    scaler = StandardScaler()
    scaler.fit(X)
    reg = LinearRegression().fit(scaler.transform(X), y)


    for i in range(prediction_length):
        length = len(voice1)
        X = [voice1[length-window_size::]]
        y_pred = reg.predict(scaler.transform(X))
        y_pred = Postprocessing.prediction_vector_pick_highest(y_pred, zero_bias=0.1)[0]


        vector = np.append(vector, y_pred)

        # Converting vector to note number
        note = Postprocessing.vector_to_note(min_note, y_pred)

        predicted_voice.append(note)
        voice1 = np.append(voice1, note)

    np.savetxt("output_complete.txt", voice1.astype(int), fmt='%s')
    np.savetxt("output_new.txt", predicted_voice, fmt='%s')

    Convert_to_MIDI.convert_to_midi("output_complete.txt", "output_complete.mid")
    Convert_to_MIDI.convert_to_midi("output_new.txt", f"output_new{tracknumber}.mid")

if __name__ == '__main__':
    for i,wsize in zip([1,2,3,4],[670, 500, 500, 600]):
        # Best so far: [650/670, 500, 680, 600]
        create_track(i, wsize)