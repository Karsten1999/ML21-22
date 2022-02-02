import numpy as np

def read_length(voice):
	"""
	:param data: the input voice

	:output data: the length (in timesteps) that each note is played
	:output data: the notes that correspond to each length
	"""
	length = np.array([])
	filtvoice = np.array([])
	for i in range(len(voice)):
		if i == 0:
			length = np.append(length, 1)
			filtvoice = np.append(filtvoice, voice[i])
		elif voice[i] != voice[i - 1]:
			length = np.append(length, 1)
			filtvoice = np.append(filtvoice, voice[i])
		else:
			length[-1] += 1
	return length.astype(int), filtvoice.astype(int)

def note_played(voice):
	"""Boolean function whether a note is played at a given timestep."""
	return voice.astype(bool)

def pitch(note, minnote, maxnote):
	"""Return pitch of a given note as shown in the paper"""
	# f major/ d minor key
	note_name = int((note - 53) % 12) # 53 is F3 note 
	chroma = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]) # chroma circle
	fifths = np.array([1, 8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6]) # circle of fifths
	radius_chroma = 1 # possible adjustments in importance of circles
	radius_fifths = 1
	maxpitch = 2 * np.log2(2 ** ((maxnote - 69) / 12) * 440)  # A4 = 440 Hz and note 69
	minpitch = 2 * np.log2(2 ** ((minnote - 69) / 12) * 440)
	chroma_x = radius_chroma * np.cos(chroma[note_name] * np.pi / 6)
	chroma_y = radius_chroma * np.sin(chroma[note_name] * np.pi / 6)
	fifths_x = radius_fifths * np.cos(fifths[note_name] * np.pi / 6)
	fifths_y = radius_fifths * np.sin(fifths[note_name] * np.pi / 6)
	freq = 2 ** ((note - 69) / 12) * 440 
	pitch = 2 * np.log2(freq) - maxpitch + (maxpitch - minpitch) / 2
	return np.array([[pitch, chroma_x, chroma_y, fifths_x, fifths_y]]).T

def find_pitches(voice):
	"""Finds pitches for all notes in a voice"""
	notelist = np.empty((5, 1))

	minnote = min(voice[voice > 0])
	maxnote = max(voice)
	for i in range(len(voice)):
		if voice[i] == 0:
			note = np.zeros((5,1))
		else:
			note = pitch(voice[i] + 8, minnote + 8, maxnote + 8) # +8 is correction to MIDI
		notelist = np.append(notelist, note, axis = 1)
	return notelist

def inverse_pitch(pitches, maxpitch, minpitch):
	"""Reverts pitch transformation into a midi note."""
	pitch = pitches[0,:]
	note_name = np.zeros(len(pitch))
	n = 69 + 12 * np.log2(1/440) + 3 * maxpitch + 3 * minpitch + 6 * pitch # inverse of pitch function
	for i in range(len(n)):
		if sum(np.abs(pitches[:,i]))>1e-10: # avoid floating point errors
			note_name[i] = n[i]
	return note_name



def find_lim_pitches(voice):
	"""Finds max and min pitches for a given voice."""
	maxnote = max(voice)
	minnote = min(voice[voice > 0])
	maxpitch = 2 * np.log2(2 ** ((maxnote - 69) / 12) * 440)  # A4 = 440 Hz and note 69
	minpitch = 2 * np.log2(2 ** ((minnote - 69) / 12) * 440)
	return maxpitch, minpitch

voice1 = np.loadtxt("../data/F.txt").T[2]
x = find_pitches(voice1)
maxp, minp = find_lim_pitches(voice1)
v = inverse_pitch(x, maxp, minp)