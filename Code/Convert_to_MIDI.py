from midiutil import MIDIFile
import numpy as np

def get_note_duration_time(data):
    """"

    """
    note_duration_time_pair = []

    duration = 0
    current_time = 0
    next_time = 0
    current_note = data[0]
    for note in data:
        # If the next note is the same as the current one we increase the duration by 1
        if note == current_note:
            # Each row is a 1/4th beat, see page 7 https://lss.fnal.gov/archive/other/print-93-0456.pdf
            duration += 1/4
        else:
            # appending the last note, since now the code will move on
            # 20 is added since a MIDI note is 20 higher than the note number
            note_duration_time_pair.append([int(current_note)+20, duration, current_time])
            duration = 1/4
            # Updating note
            current_note = note
            # Updating time
            current_time = next_time

        next_time += 1/4
    return note_duration_time_pair



def convert_to_midi(filename: str):
    """"
    :param filename : name of the file to be converted to midi, has to be a txt containing one or more columns
    """

    # Reading in data and transposing it so each column is a voice
    data = np.loadtxt(filename).T

    # The txt file is given as a sequence of tones, however we want a sequence of tones and how long they last,
    # Therefore we have to loop over each column to find this out

    if data.ndim == 1:
        note_duration_time_pair = get_note_duration_time(data)
    elif data.ndim == 2:
        note_duration_time_pair = []
        for column in data:
            note_duration_pair_time_voice = get_note_duration_time(column)
            note_duration_time_pair.append(note_duration_pair_time_voice)

    # General information for midi
    channel = 0
    volume = 100
    # Amount of voices
    print(note_duration_time_pair)
    if data.ndim == 1:
        voices = 1
    elif data.ndim == 2:
        voices = data.shape[0]
    # Checking the dimension of the dat data to see if we have to convert multiple voices
    if voices == 1:
        track = 0
        MyMIDI = MIDIFile(1)
        for note, duration, time in note_duration_time_pair:
            # Only actually add notes that are non-zero (or non-20 since we added 20)
            if note>20:
                MyMIDI.addNote(track, channel, note, time, duration, volume)
    elif voices > 1:
        MyMIDI = MIDIFile(voices)
        tracks = range(0, voices)
        for track in tracks:
            for note, duration, time in note_duration_time_pair[track]:
                # Only actually add notes that are non-zero (or non-20 since we added 20)
                if note > 20:
                    MyMIDI.addNote(track, channel, int(note), time, duration, volume)
    if MyMIDI:
        with open("output.mid", "wb") as out:
            MyMIDI.writeFile(out)


if __name__ == "__main__":
    convert_to_midi("F.txt")
