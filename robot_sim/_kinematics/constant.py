from enum import Enum


class JointType(Enum):
    REVOLUTE = 1
    PRISMATIC = 2


# declare constants
TCP_INDICATOR_STICK_RADIUS = .0025
FRAME_STICK_RADIUS = .0025
FRAME_STICK_LENGTH_LONG = .075
FRAME_STICK_LENGTH_SHORT = .0625
FRAME_L2R_RATIO_FOR_SM = 25  # frame stick axis_length vs stick radius for stick model
FRAME_L2R_RATIO_FOR_M = 30  # frame stick axis_length vs stick radius for mesh model
ANCHOR_BALL_RADIUS = .0049
JOINT_RADIUS = .0035
