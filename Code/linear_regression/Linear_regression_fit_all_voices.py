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
from multiprocessing import Pool, freeze_support


def get_score(voice, vector, i, lengths):
    X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice, vector, window_size=i, output_size=4)
    reg = LinearRegression().fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    yp = []
    for y in y_pred:
        f = Postprocessing.prediction_vector_pick_highest
        flaty = [f(y[0:lengths[0]], zero_bias=0.1), f(y[lengths[0]:lengths[1]+lengths[0]], zero_bias=0.1),
                 f(y[lengths[0]+lengths[1]:lengths[0]+lengths[1]+lengths[2]], zero_bias=0.1),
                 f(y[lengths[0]+lengths[1]+lengths[2]::], zero_bias=0.1)]
        flaty = np.array(flaty).flatten()

        yp.append(flaty)


    return accuracy_score(y_test, y_pred)

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

    lengths = [len(vector[i]) for i in range(4)]

    score = []
    scorestd = []
    difference = []

    maxwindowsize = 700
    window = range(4, maxwindowsize * 4, 4)
    tot = len(window)
    k = 0

    # Multiprocessing
    freeze_support()
    pool = Pool()

    for i in window:
        tempscore = pool.starmap(get_score, [[newvoice, vector, i, lengths] for k in range(8)])

        score.append(np.mean(tempscore))
        scorestd.append(np.std(tempscore))
        k += 1
        print("Done:", k / tot)


    score = np.array(score)
    scorestd = np.array(scorestd)

    plt.plot(window, score)
    plt.fill_between(window, score - scorestd, score + scorestd, alpha = 0.2)
    plt.title(r"Accuracy score of linear regression vs windows size for multiple voices")
    plt.xlabel("Window size")
    plt.ylabel(r"Accuracy score")
    plt.grid()
    plt.savefig("regression_score_multiple_accuracy.pdf")
    plt.show()

