from ..preprocessing import Preprocessing
from ..postprocessing import Postprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

voice1 = np.loadtxt("../data/F.txt").T[0]
vector = Preprocessing.Transform_into_vector(voice1)
score = []
difference = []

window = range(8,256,8)
tot = len(window)
k = 0

scaler = StandardScaler()

for i in window:
    tempscore = []
    for p in range(50):
        X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice1, vector, window_size=i)
        scaler.fit(X_train)

        X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

        reg = LinearRegression().fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        y_pred = Postprocessing.prediction_vector_pick_highest(y_pred, zero_bias=0.1)

        tempscore.append(accuracy_score(y_test, y_pred))
    score.append(np.mean(tempscore))
    k += 1
    print("Done:", k / tot)

plt.plot(window, score)
plt.xlabel("Window size")
plt.ylabel("Score")
plt.grid()
plt.savefig("regression_score.pdf")
plt.show()

