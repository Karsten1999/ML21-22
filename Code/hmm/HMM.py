import sys
import os
sys.path.append(os.path.abspath('../preprocessing'))
sys.path.append(os.path.abspath('../general'))

import numpy as np
from hmmlearn import hmm # Hidden Markov Model package
from sklearn import preprocessing as pre

import Convert_to_MIDI
import Note_representation

# Contrapunctus XIV is split in 3 parts, 
# this index is what I identify as the start of the third part
# upon listening to the piece. Subtracted 1 since arrays start at 0.
first_note_of_3rd_section = 3095




def hmm_fit(voice, nSamp, components):
	"""Return nSamp fitted note-length pairs using a HMM with n_components = components."""
	# LabelEncoder transformations
	le = pre.LabelEncoder()
	le.fit(voice)
	newvoice=le.transform(voice)

	#encoding of lengths and note values into 1 int
	lengths, notes = Note_representation.read_length(newvoice.reshape(-1, 1))
	a = max(notes)
	notevec = (a * lengths + notes)

	# another LabelEncoder transformation
	le2 = pre.LabelEncoder()
	le2.fit(notevec)
	transnotes = le2.transform(notevec).reshape(-1, 1)

	# Actual model
	model = hmm.MultinomialHMM(n_components = components)
	model.fit(transnotes)
	x, s = model.sample(nSamp)

	# Inverse transformation
	fitnotes = le2.inverse_transform(np.ravel(x))
	flengths = np.floor(fitnotes / a)
	fnotes = le.inverse_transform(fitnotes % a)
	results = np.append(fnotes[:, np.newaxis], flengths[:, np.newaxis], axis=1)
	return results

def output_midi(res, voice, filename):
	"""Transform an original voice and outputs into a MIDI file."""
	f = voice
	for i in range(len(res)):
		for j in range(int(res[i, 1])):
			f = np.append(f, res[i, 0])
	np.savetxt(filename + ".txt", f, fmt = '%i')
	Convert_to_MIDI.convert_to_midi(filename + ".txt", filename + ".mid")

# reading data and creating MIDI files
voice = np.loadtxt("../data/F.txt").T
for i in range(len(voice)):
	fit4 = hmm_fit(voice[i], 100, 4)
	fit7 = hmm_fit(voice[i], 100, 7)
	output_midi(fit4, voice[i], "4comps{:}".format(i))
	output_midi(fit7, voice[i], "7comps{:}".format(i))