import Preprocessing
import Postprocessing
import Convert_to_MIDI
import numpy as np
from sklearn.linear_model import LinearRegression

voice1 = np.loadtxt("F.txt").T[0][3408::]
predicted_voice = []
vector = Preprocessing.Transform_into_vector(voice1)
print(len(vector[0]))

min_note = min(voice1)

# Parameter to set length of prediction
prediction_length = 128
# Parameter to set windows size
window_size = 32

X, y = Preprocessing.Split_rolling_window(voice1, vector, window_size=window_size, train=False)
reg = LinearRegression().fit(X, y)

#X_predict = [voice1[len(voice1)-window_size::]]
#test = Postprocessing.prediction_vector_traverse(X_predict, reg, min_note, prediction_length=prediction_length
#                                                 , limit_chance=0.2)

#print(test)

for i in range(prediction_length):
    length = len(voice1)
    X = [voice1[length-window_size::]]
    y_pred = reg.predict(X)
    y_pred = Postprocessing.prediction_vector_pick_stochastically(y_pred, min_chance=0.1)[0]


    vector = np.append(vector, y_pred)

    # Converting vector to note number
    note = Postprocessing.vector_to_note(min_note, y_pred)

    predicted_voice.append(note)
    voice1 = np.append(voice1, note)

np.savetxt("output_complete.txt", voice1.astype(int), fmt='%s')
np.savetxt("output_new.txt", predicted_voice, fmt='%s')

Convert_to_MIDI.convert_to_midi("output_complete.txt", "output_complete.mid")
Convert_to_MIDI.convert_to_midi("output_new.txt", "output_new.mid")
