import os

import numpy as np
import matplotlib.pyplot as plt


class HitStats:
    def __init__(self, map_shape, record_every=100):
        self._hit_map = np.zeros(map_shape, dtype=bool)
        self._access_count = 0
        self._hit_count = 0

        self._record_every = record_every
        self.hit_values = []
        self.access_values = []
        
    def to_file(self, save_dir='.'):
        results = np.array([self.access_values, self.hit_values])
        save_fn = os.path.join(save_dir, f'hit_stats.npy')
        np.save(save_fn, results)

    def record_access(self, indices):
        self._access_count += 1

        # update map and hit count
        if self._hit_map[indices] == 0:
            self._hit_map[indices] = 1
            self._hit_count += 1

        # check if we need to record
        if self._access_count % self._record_every == 0:
            self.access_values.append(self._access_count)
            self.hit_values.append(self._hit_count)

    def show_hit_stats(self):
        fig, ax = plt.subplots()
        ax.plot(self.access_values, self.hit_values)
        ax.set_ylabel('cumulative number of new map elements')
        ax.set_xlabel('number of samples')
        plt.show()
