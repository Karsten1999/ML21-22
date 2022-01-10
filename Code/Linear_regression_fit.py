import Preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np
from matplotlib import pyplot as plt

voice1 = np.loadtxt("F.txt").T[0]
vector = Preprocessing.Transform_into_vector(voice1)
score = []
difference = []

window = range(1,128,1)
tot = len(window)
k = 0
for i in window:
    temp_score = []
    for j in range(49):
        X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice1, vector, window_size=i)
        reg = LinearRegression().fit(X_train, y_train)

        y_pred = reg.predict(X_test)

        y_prediction = []
        for yp in y_pred:
            yp[yp<max(yp)]=0
            yp[yp==max(yp)]=1
            y_prediction.append(yp)
        temp_score.append(r2_score(y_test, y_prediction))
    score.append(np.mean(temp_score))
    k+=1
    print("Done", k / tot)

plt.plot(window, score)
plt.xlabel("Window size")
plt.ylabel("Score")
plt.grid()
plt.savefig("regression_score.pdf")
plt.show()

