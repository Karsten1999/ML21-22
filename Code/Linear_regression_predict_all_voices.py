import Preprocessing
import Postprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, accuracy_score, f1_score, adjusted_mutual_info_score
from tqdm import tqdm

voice1 = np.loadtxt("F.txt").T[0]
voice2 = np.loadtxt("F.txt").T[1]
voice3 = np.loadtxt("F.txt").T[2]
voice4 = np.loadtxt("F.txt").T[3]

vector1 = Preprocessing.Transform_into_vector(voice1)
vector2 = Preprocessing.Transform_into_vector(voice2)
vector3 = Preprocessing.Transform_into_vector(voice3)
vector4 = Preprocessing.Transform_into_vector(voice4)

# save the lengths of the individual vectors into array
vector_lengths = [len(vector1[0]), len(vector2[0]), len(vector3[0]), len(vector4[0])]

# paste into big vector
vector = np.hstack([vector1, vector2, vector3, vector4])
# paste voice into large vector of voices, alternating
voice = [item for sublist in zip(voice1, voice2, voice3, voice4) for item in sublist]

score = []
difference = []

window = range(1,128*4,1)
tot = len(window)

for i in tqdm(window):
    temp_score = []
    for j in range(4):
        X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice, vector, window_size=i)
        reg = LinearRegression().fit(X_train, y_train)

        y_probs_pred = reg.predict(X_test)

        m = 0
        #separate y_pred into individual vectors and pick highest value
        for n in vector_lengths:
            y_pred_i = Postprocessing.prediction_vector_pick_stochastically(y_probs_pred[:, m:m+n])
#             y_pred_i = Postprocessing.prediction_vector_pick_highest(y_probs_pred[:, m:m+n])

            if m == 0:
                y_pred = y_pred_i
            else:
                y_pred = np.hstack([y_pred, y_pred_i])
                
            m += n

        temp_score.append(r2_score(y_test, y_pred))
    score.append(np.mean(temp_score))

plt.plot(window, score)
plt.xlabel("Window size")
plt.ylabel("Score")
plt.grid()
# plt.savefig("regression_score.pdf")
plt.show()

