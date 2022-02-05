import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
import Preprocessing
import Postprocessing
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool, freeze_support


def get_score(voice, vector, i):
    X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice, vector, window_size=i)

    reg = LinearRegression().fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    y_pred = Postprocessing.prediction_vector_pick_highest(y_pred, zero_bias=0.1)

    return r2_score(y_test, y_pred)

if __name__ == '__main__':
    voice1 = np.loadtxt("../data/F.txt").T[0][3095::]
    vector = Preprocessing.Transform_into_vector(voice1)
    score = []
    scorestd = []
    difference = []

    window = range(1, 700, 1)
    tot = len(window)
    k = 0

    scaler = StandardScaler()

    # Multiprocessing
    freeze_support()
    pool = Pool()

    for i in window:
        tempscore = pool.starmap(get_score, [[voice1, vector, i] for k in range(8)])

        score.append(np.mean(tempscore))
        scorestd.append(np.std(tempscore))
        k += 1
        print("Done:", k / tot)


    score = np.array(score)
    scorestd = np.array(scorestd)

    plt.plot(window, score)
    plt.fill_between(window, score - scorestd, score + scorestd, alpha = 0.2)
    plt.title(r"$R^2$ score of linear regression vs windows size")
    plt.xlabel("Window size")
    plt.ylabel(r"$R^2$ score")
    plt.grid()
    plt.savefig("regression_score_R2.pdf")
    plt.show()

