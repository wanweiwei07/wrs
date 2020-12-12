'''
Logger YuMi motions for debug purposes. 
Author: Jacky Liang
'''

import pickle
import copy

class YuMiMotionLogger:

    _SUFFIX = '.yml'

    def __init__(self):
        self.reset_log()

    def reset_log(self):
        self._times = []
        self._expected = []
        self._actual = []

    def append_time(self, time):
        self._times.append(time)

    def append_expected(self, expected):
        self._expected.append(expected)

    def append_actual(self, actual):
        self._actual.append(actual)
        
    @property
    def times(self):
        return copy.copy(self._times)
        
    @property
    def expected(self):
        return copy.copy(self._expected)
        
    @property
    def actual(self):
        return copy.copy(self._actual)

    def flush_to_file(self, filename):
        if not filename.endswith(YuMiMotionLogger._SUFFIX):
            raise Exception("Can only flush to {0} files!".format(YuMiMotionLogger._SUFFIX))
        with open(filename, 'w') as file:
            pickle.dump(self, file)
        self.reset_log()

    @staticmethod
    def load(filename):
        if not filename.endswith(YuMiMotionLogger._SUFFIX):
            raise Exception("Can only load from {0} files!".format(YuMiMotionLogger._SUFFIX))
        with open(filename, 'r') as file:
            return pickle.load(file)