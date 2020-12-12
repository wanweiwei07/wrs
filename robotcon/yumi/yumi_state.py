'''
State Encapsulation for YuMi robot
Author: Jacky
'''
import numpy as np

class YuMiState:
    """ Object that encapsulates a yumi arm joint angle configuration.
    """

    NUM_JOINTS = 7
    NAME = "YuMi"

    def __init__(self, vals = [0] * NUM_JOINTS):
        for i, val in enumerate(vals):
            setattr(self, '_joint{0}'.format(i+1), val)

    def __str__(self):
        return str(self.joints)

    def __repr__(self):
        return "YuMiState({0})".format(self.joints)

    @property
    def joints(self):
        joints = [getattr(self, '_joints{0}'.format(i)) for i in range(YuMiState.NUM_JOINTS)]
        return joints

    @property
    def in_radians(self):
        return [np.pi / 180.0 * t for t in self.joints]

    @property
    def in_degrees(self):
        return self.joints

    @property
    def joint1(self):
        return self._joint1

    @joint1.setter
    def joint1(self, val):
        self._joint1 = val

    @property
    def joint2(self):
        return self._joint2

    @joint2.setter
    def joint2(self, val):
        self._joint2 = val

    @property
    def joint3(self):
        return self._joint3

    @joint3.setter
    def joint3(self, val):
        self._joint3 = val

    @property
    def joint4(self):
        return self._joint4

    @joint4.setter
    def joint4(self, val):
        self._joint4 = val

    @property
    def joint5(self):
        return self._joint5

    @joint5.setter
    def joint5(self, val):
        self._joint5 = val

    @property
    def joint6(self):
        return self._joint6

    @joint6.setter
    def joint6(self, val):
        self._joint6 = val

    @property
    def joint7(self):
        return self._joint7

    @joint7.setter
    def joint7(self, val):
        self._joint7 = val

    @property
    def joints(self):
        joints = [getattr(self, 'joint{0}'.format(i+1)) for i in range(YuMiState.NUM_JOINTS)]
        return joints

    def copy(self):
        return YuMiState(self.joints)

    def mirror(self):
        return YuMiState([
            self.joint1 * -1,
            self.joint2,
            self.joint3,
            self.joint4 * -1,
            self.joint5,
            self.joint6,
            self.joint7 * -1
        ])

    def __str__(self):
        return str(self.joints)

    def __repr__(self):
        return "YuMiState({0})".format(str(self.joints))

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        return False

    def __ne__(self, other):
        if isinstance(other, self.__class__):
            return not self.__eq__(other)
        return NotImplemented

    def __hash__(self):
        return hash(tuple(sorted(self.__dict__.items())))
