from enum import Enum


class JntType(Enum):
    REVOLUTE = 1
    PRISMATIC = 2


# declare constants
TCP_INDICATOR_STICK_RADIUS = .0025
FRAME_STICK_RADIUS = .0012
FRAME_STICK_LENGTH_LONG = .075
FRAME_STICK_LENGTH_MEDIUM = .0625
FRAME_STICK_LENGTH_SHORT = .0375
FRAME_L2R_RATIO_FOR_SM = 25  # frame stick axis_length vs stick radius for stick model
FRAME_L2R_RATIO_FOR_M = 30  # frame stick axis_length vs stick radius for mesh model
ROTAX_STICK_LENGTH = .0875
JNT_RADIUS = .01
ANCHOR_RATIO = 1.2
PRISMATIC_RATIO = 1.2
LNK_STICK_RADIUS = .0072
