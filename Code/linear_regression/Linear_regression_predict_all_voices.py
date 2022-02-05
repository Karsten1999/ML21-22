import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
sys.path.append(os.path.abspath('../general'))
import Preprocessing
import Postprocessing
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
import Convert_to_MIDI

if __name__ == '__main__':
    voices = np.loadtxt("../data/F.txt").T[:,3095::]
    newvoice = []

    # Here we are creating a new voice which is all voices combined after each other, so if we had [1,2] [3,4] [5,6]
    # and [7,8] and voices at first we now get [1, 3, 5, 7, 2, 4, 6, 8]

    for a, b, c, d in zip(voices[0],voices[1],voices[2],voices[3]):
        newvoice.append(a)
        newvoice.append(b)
        newvoice.append(c)
        newvoice.append(d)

    newvoice = np.array(newvoice)
    vector = Preprocessing.Transform_into_vector(newvoice)

    min_note = min(newvoice)

    lengths = [len(vector[i]) for i in range(4)]

    # Outcome parameters
    window_size = 2600
    prediction_length = 256

    # Training the model
    X, y = Preprocessing.Split_rolling_window(newvoice, vector, window_size=window_size, train=False, output_size=4)
    reg = LinearRegression().fit(X, y)

    # New music generated
    predicted_voice = []

    for i in range(0,prediction_length,4):
        length = len(newvoice)
        X = [newvoice[length - window_size::]]

        y_pred = reg.predict(X)[0]

        # Transforming the vector
        f = Postprocessing.prediction_vector_pick_highest
        newvoice1 = f(y_pred[0:lengths[0]], zero_bias=0.1)
        newvoice2 = f(y_pred[lengths[0]:lengths[1] + lengths[0]], zero_bias=0.1)
        newvoice3 = f(y_pred[lengths[0] + lengths[1]:lengths[0] + lengths[1] + lengths[2]], zero_bias=0.1)
        newvoice4 = f(y_pred[lengths[0] + lengths[1] + lengths[2]::], zero_bias=0.1)

        for newnote in [newvoice1, newvoice2, newvoice3, newvoice4]:
            note = Postprocessing.vector_to_note(min_note, newnote)
            predicted_voice.append(note)
            newvoice = np.append(newvoice, note)



    # Now we have to transform the data back into 4 separate vectors

    new = [[], [], [], []]

    for i in range(int(len(predicted_voice)/4)):
        new[0].append(predicted_voice[i])
        new[1].append(predicted_voice[i+1])
        new[2].append(predicted_voice[i+2])
        new[3].append(predicted_voice[i+3])
    old_and_new = [[], [], [], []]
    for i in range(int(len(newvoice)/4)):
        old_and_new[0].append(newvoice[i])
        old_and_new[1].append(newvoice[i+1])
        old_and_new[2].append(newvoice[i+2])
        old_and_new[3].append(newvoice[i+3])

    np.savetxt("4voices_output_complete.txt", np.array(old_and_new).T, fmt='%s')
    np.savetxt("4voices_output_new.txt", np.array(new).T, fmt='%s')

    Convert_to_MIDI.convert_to_midi("4voices_output_complete.txt", "4voices_output_complete.mid")
    Convert_to_MIDI.convert_to_midi("4voices_output_new.txt", "4voices_output_new.mid")