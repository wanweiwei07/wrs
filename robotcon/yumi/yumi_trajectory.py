'''
Trajectory encapsulation for YuMi robot
Author: Jeff Mahler
'''
import numpy as np

class YuMiTrajectory:
    def __init__(self, states):
        self.states = states
        self._index = 0

    def state(self, i):
        return self.states[i]

    @property
    def length(self):
        return len(self.states)

    def __iter__(self):
        self._index = 0
        return self

    def next(self):
        if self._index >= self.length:
            raise StopIteration
        else:
            cur_state = self.state(self._index)
            self._index += 1
            return cur_state
