import os
from scipy.sparse import *
import random

input_folder = '../../sorted_midi_files/bar_vectors/rock/'

class RockBars():
    def __init__(self):
        self.num_files = len([f for f in os.listdir(input_folder)]) - 1
        print("loaded folder, batch contains: " + str(self.num_files) + " files")

    def get_minibatch(self, mb_size):
        f_names = random.sample(range(0, self.num_files), mb_size)
        mat = load_npz(input_folder + str(f_names[0]) + '.npz')
        for f in f_names[1:]:
            vec = load_npz(input_folder + str(f) + '.npz')
            mat = vstack([mat, vec])
        return mat.toarray()
