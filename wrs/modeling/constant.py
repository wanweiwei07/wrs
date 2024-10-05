from enum import Enum
from panda3d.core import BitMask32


class CDMeshType(Enum):
    AABB = 1
    OBB = 2
    CONVEX_HULL = 3
    CYLINDER = 4
    DEFAULT = 5  # triangle mesh


class CDPrimType(Enum):
    AABB = 1
    OBB = 2
    CAPSULE = 3
    CYLINDER = 4
    SURFACE_BALLS = 5
    POINT_CLOUD = 6
    USER_DEFINED = 7