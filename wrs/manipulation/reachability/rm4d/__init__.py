"""
Official Implementation of  the paper "RM4D: A Combined Reachability and Inverse Reachability Map
for Common 6-/7-axis Robot Arms by Dimensionality Reduction to 4D"
Copy from github: https://github.com/mrudorfer/rm4d
The RM4D is adapted and extended for the WRS.
"""

from .reachability_map import ReachabilityMap4D
from .construction import JointSpaceConstructor
from .map_stats import HitStats
