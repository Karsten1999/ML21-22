from hmmlearn import hmm
import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
sys.path.append(os.path.abspath('../general'))
import Note_representation
import numpy as np
import matplotlib.pyplot as plt
import Convert_to_MIDI
import Preprocessing
import Postprocessing



voice1 = np.loadtxt("../data/F.txt").T[0][3095::]
maxp, minp = Note_representation.find_lim_pitches(voice1)

voice1 = Note_representation.find_pitches(voice1).T

model = hmm.GaussianHMM(n_components=9)
model.fit(voice1)

newvoice = model.sample(256)[0]
newvoice = Note_representation.inverse_pitch(newvoice.T, maxp, minp)
finalvoice = [int(round(i)) for i in newvoice]

#np.savetxt("GHMM_complete.txt", voice1.astype(int), fmt='%s')
np.savetxt("GHMM_new.txt", finalvoice, fmt='%s')
Convert_to_MIDI.convert_to_midi("GHMM_new.txt", "GHMM_new.mid")


""""
voice1 = np.loadtxt("../data/F.txt").T[0][3095::]

minnote = np.min(voice1[voice1!=0])

voice1 = Preprocessing.Transform_into_vector(voice1)

model = hmm.GaussianHMM(n_components=13)
model.fit(voice1)

newvoice = model.sample(256)[0]
newvoice = Postprocessing.prediction_vector_pick_highest(newvoice)

finalvoice = []
for note in newvoice:
    notenumber = Postprocessing.vector_to_note(minnote, note)
    finalvoice.append(notenumber)

np.savetxt("GHMM_new.txt", finalvoice, fmt='%s')
Convert_to_MIDI.convert_to_midi("GHMM_new.txt", "GHMM_new.mid")
"""