from enum import Enum
from panda3d.core import BitMask32


class CDMType(Enum):
    AABB = 1
    OBB = 2
    CONVEX_HULL = 3
    CYLINDER = 4
    DEFAULT = 5  # triangle mesh


class CDPType(Enum):
    BOX = 1
    CAPSULE = 2
    CYLINDER = 3
    SURFACE_BALLS = 4
    POINT_CLOUD = 5
    USER_DEFINED = 6