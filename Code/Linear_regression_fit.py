import Preprocessing
from sklearn.linear_model import LinearRegression
import numpy as np
from matplotlib import pyplot as plt

voice1 = np.loadtxt("F.txt").T[0]
vector = Preprocessing.Transform_into_vector(voice1)
score = []

window = range(1,128,1)
tot = len(window)
k = 0
for i in window:
    X_train, X_test, y_train, y_test = Preprocessing.Split_rolling_window(voice1, vector, window_size=i)
    reg = LinearRegression().fit(X_train, y_train)
    score.append(reg.score(X_test, y_test))
    k+=1
    print("Done", k / tot)

plt.plot(window, score)
plt.xlabel("Window size")
plt.ylabel("Score")
plt.grid()
plt.savefig("regression_score.pdf")
plt.show()

