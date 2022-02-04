from hmmlearn import hmm
import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
import Note_representation
import numpy as np
import matplotlib.pyplot as plt


def parametersearch(train, test, end_n = 18):
    """"
    Find best parameters for the model and returning the scores for the parameters
    """
    # Data for storing the best parameters in the order model, n_components, covariance_type
    bestscore = 0
    bestparam = {}

    # Data for plotting
    allscores = []

    all_n = range(1, end_n)
    for cov in ['spherical', 'diag']:
        for n in all_n:
            # Training the model
            model = hmm.GaussianHMM(n_components=n,covariance_type=cov)
            model.fit(train)

            # Scoring the model
            score = model.score(test)
            allscores.append(score)
            if score > bestscore:
                bestscore = score
                bestparam = {'model': hmm.GaussianHMM, 'n_components': n, 'covariance_type': cov, 'score': score}
    return bestparam, allscores


def plot_score(scores, end_n):
    """"
    Function to plot the bar charts for the scores
    """
    # Extracting the corresponding scores
    sphscore = scores[0:end_n - 1]
    diagscore = scores[end_n - 1::]

    all_n = range(1, end_n)

    fig, ax = plt.subplots()

    # Setting up the bars
    width = 0.4
    x = np.arange(len(all_n))

    ax.bar(x - 1 / 2 * width, sphscore, width, label='Spherical')
    ax.bar(x + 1 / 2 * width, diagscore, width, label='Diagonal')

    # Generic graph stuff
    ax.set_ylabel('Scores')
    ax.set_xlabel('Number of states')
    ax.set_title('Log-likelihood of GHMM for different hyperparameters: 1 voice')
    ax.set_xticks(x)
    ax.set_xticklabels(x + 1)
    ax.grid()
    ax.legend()

    fig.tight_layout()
    fig.savefig("Hyperparameter_search_GHMM_1voice.pdf")

    plt.show()


# Loading and transforming the data
voice1 = np.loadtxt("../data/F.txt").T[0][3095::]
voice1 = Note_representation.find_pitches(voice1).T


split = int(len(voice1)*0.8)
traindata, testdata = voice1[0:split], voice1[split::]

# Finding best paramaters for HMM
bestparam, allscores = parametersearch(traindata, testdata)
print(bestparam)

allscores = np.array(allscores)

allscores[allscores<-5000]=-5000

plot_score(allscores, 18)




