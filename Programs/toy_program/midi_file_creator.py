import midi
from scipy.sparse import *
import numpy as np
import matplotlib.pyplot as plt

instruments = [2, 27, 0, 56] # the instruments to convert the channel values to
min_val = 0.1 # the minimum value for a note to be considered "off" as the array is floats

def get_vectors(array, channel):
    # takes an (n * 128) x 128 array and returns a list of vectors
    vectors = []
    array = array.transpose() # give 128 x (n*128) array
    for note_val, note in enumerate(array):
        # iterate through all notes and find start and run time in ticks
        note_found = False
        for t_step, is_active in enumerate(note):
            if is_active > min_val:
                if not note_found:
                    # a note start has been found
                    note_found = True
                    run_time = 0
                    vectors.append((channel, note_val, t_step, 1))
            else:
                if note_found:
                    note_found = False
                    vectors.append((channel, note_val, t_step, 0))
        if note_found:
            vectors.append((channel, note_val, t_step, 0))

    vectors.sort(key=lambda x: x[2])
    # returns vectors of the form (channel, note_val, time_step, note_on/note_off)
    return vectors


def reshape_array(array):
    n = array.shape[0]
    output_array = np.zeros((4, n*128, 128))

    for a in range(0, n):
        for b in range(0, 4):
            for c in range(0, 128):
                for d in range(0, 128):
                    output_array[b][(a * 128) + c][d] = array[a][(b * 128 * 128) + (c * 128) + d]

    return output_array


def convert_to_midi(array, output_file_location):
    # takes an n x 65536 array, and outputs a midi file to the output location
    # first reshape to 4 x (n*128) x 128 array
    array = reshape_array(array)

    # pass to get_vectors
    track_vectors = []
    for ind, track in enumerate(array):
        if ind == 2:
            channel = 9
        else:
            channel = ind
        track_vectors.append(get_vectors(track, channel))

    event_vectors = track_vectors[0] + track_vectors[1] + track_vectors[2] + track_vectors[3]

    event_vectors.sort(key=lambda x: x[2])

    # create midi file
    pattern = midi.Pattern(resolution=32)
    for ind, vectors in enumerate(track_vectors):
        track = midi.Track()
        pattern.append(track)
        if ind == 2:
            ch = 9
        else:
            ch = ind
        track.append(midi.ProgramChangeEvent(tick=0, channel = ch, data=[instruments[ind]]))
        last_event_tick = 0
        for event in vectors:
            if event[3] == 1:
                track.append(midi.NoteOnEvent(tick=event[2] - last_event_tick, channel=event[0], data=[event[1], 50]))
            else:
                track.append(midi.NoteOffEvent(tick=event[2] - last_event_tick, channel=event[0], data=[event[1], 50]))
            last_event_tick = event[2]
        track.append(midi.EndOfTrackEvent(tick=1))

    print(pattern)
    midi.write_midifile(output_file_location, pattern)
    return

if __name__ == '__main__':
    #array = load_npz('../../sorted_midi_files/array_files/rock/Live_And_Let_Die.npz')
    array = load_npz('./mb_16/vector/99.npz')
    array = array.toarray()
    array = array.astype(float)
    convert_to_midi(array, './it99.mid')


