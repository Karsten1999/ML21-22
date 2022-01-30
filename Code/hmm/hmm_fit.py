from hmmlearn import hmm
from ..preprocessing import Note_representation
import numpy as np

voice1 = np.loadtxt("../data/F.txt").T[0]

voice1 = Note_representation.find_pitches(voice1)

print(voice1)