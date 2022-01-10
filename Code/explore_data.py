import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from Convert_to_MIDI import get_note_duration_time
from pandas.plotting import autocorrelation_plot

data = np.loadtxt("F.txt").T

temp = get_note_duration_time(data[0])

x = []
y = []
z = []

for t in temp:
    x.append(t[0])
    y.append(t[1])
    z.append(t[2])

x = np.array(x, dtype = float)
y = np.array(y, dtype = float)
z = np.array(z, dtype = float)

print(len(temp))

if False:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(x[x>20],y[x>20])
    ax.set_xlabel('Note')
    ax.set_ylabel('Duration')
    ax.grid()
    plt.show()


if False:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[x>20],y[x>20],z[x>20])
    ax.set_xlabel('Note')
    ax.set_ylabel('Duration')
    ax.set_zlabel('Time')
    ax.grid()
    plt.show()

if False:
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.acorr(x=x, maxlags=100)
    ax.grid()
    plt.show()

if False:
    autocorrelation_plot(x)
    plt.show()