from hmmlearn import hmm
import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
import Note_representation
import numpy as np

# Loading and transforming the data
voice1 = np.loadtxt("../data/F.txt").T[0][2000::]
voice1 = Note_representation.find_pitches(voice1).T


split = int(len(voice1)*0.8)
traindata, testdata = voice1[0:split], voice1[split::]

# Data for storing the best parameters in the order model, n_components, covariance_type
bestscore = 0

# Finding best paramaters for HMM
for cov in ['spherical', 'diag', 'full', 'tied']:
    for n in range(1,21):
        # Training the model
        model = hmm.GaussianHMM(n_components=n, n_iter=10000, covariance_type=cov)
        model.fit(traindata)

        # Scoring the model
        score = model.score(testdata)
        if score > bestscore:
            bestscore = score
            bestparam = {'model': hmm.GaussianHMM, 'n_components': n, 'covariance_type': cov, 'score': score}
print(bestparam)
""""
# Finding best paramaters for GMMHMM
for cov in ['spherical', 'diag', 'full', 'tied']:
    for n_mix in range(3,10):
        for n in range(n_mix,21):
            # Training the model
            model = hmm.GMMHMM(n_components=n, n_iter=10000, covariance_type=cov, n_mix=n_mix)
            model.fit(traindata)

            # Scoring the model
            score = model.score(testdata)
            print(n_mix, n, score)
            if score > bestscore:
                bestscore = score
                bestparam = {'model': hmm.GaussianHMM, 'n_components': n, 'covariance_type': cov, 'n_mix': n_mix,
                             'score': score}
"""


#print(model.sample(100))

#state_sequence = model.predict(voice1)
#prob_next_step = model.transmat_[state_sequence[-1], :]

#print(state_sequence[-1])
#print(prob_next_step)


