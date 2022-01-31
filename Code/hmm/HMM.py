import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../postprocessing'))
sys.path.append(os.path.abspath('../general'))
import numpy as np
from hmmlearn import hmm # Hidden Markov Model package
import Note_representation
from sklearn import preprocessing as pre
import Convert_to_MIDI
#from Preprocessing import Transform_into_vector

# Contrapunctus XIV is split in 3 parts, 
# this is what I identify as the start of the third part
# upon listening to the piece. Subtracted 1 since arrays start at 0.
first_note_of_3rd_section = 3095

voice1 = np.loadtxt("../data/F.txt").T[0]

# ugly transformations
le = pre.LabelEncoder()
le.fit(voice1)
newvoice=le.transform(voice1)
lengths, notes=Note_representation.read_length(newvoice.reshape(-1, 1))

# encoding of lengths and note values into 1 int
a = max(notes)
notevec = (a * lengths + notes)

le2=pre.LabelEncoder()
le2.fit(notevec)
transnotes=le2.transform(notevec).reshape(-1, 1)

# Actual model
model = hmm.MultinomialHMM(n_components = 4)
model.fit(transnotes)
x,s=model.sample(100)
print(np.ravel(x))
print(s)

# Inverse transformation
fitnotes = le2.inverse_transform(np.ravel(x))
flengths = np.floor(fitnotes / a)
fnotes = le.inverse_transform(fitnotes % a)
results = np.append(fnotes[:, np.newaxis], flengths[:, np.newaxis], axis=1)
print(results.shape)

fitvoice=voice1
for i in range(len(results)):
	for j in range(int(results[i,1])):
		fitvoice=np.append(fitvoice,results[i,0])
np.savetxt("fitoutput.txt", fitvoice, fmt='%i')
Convert_to_MIDI.convert_to_midi("fitoutput.txt", "fitoutput.mid")