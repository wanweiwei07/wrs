from sympy.physics.quantum.matrixutils import numpy_to_sympy

from wrs import wd, mcm, rm
from sympy import *
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta

base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
mcm.mgm.gen_frame().attach_to(base)
arm = cbta.CobottaArm(enable_cc=True)

x0, x1, x2, x3, x4, x5 = symbols('x0 x1 x2 x3 x4 x5')
tp0, tp1, tp2 = symbols('tp0 tp1 tp2')
tR00, tR01, tR02, tR10, tR11, tR12, tR20, tR21, tR22 = symbols('tR00 tR01 tR02 tR10 tR11 tR12 tR20 tR21 tR22')

tp = Matrix([tp0, tp1, tp2])
tR = Matrix([[tR00, tR01, tR02], [tR10, tR11, tR12], [tR20, tR21, tR22]])

k0 = Matrix(arm.jnts[0].loc_motion_ax)
k1 = Matrix(arm.jnts[1].loc_motion_ax)
k2 = Matrix(arm.jnts[2].loc_motion_ax)
k3 = Matrix(arm.jnts[3].loc_motion_ax)
k4 = Matrix(arm.jnts[4].loc_motion_ax)
k5 = Matrix(arm.jnts[5].loc_motion_ax)

p01 = Matrix(arm.jnts[0].loc_pos)
p12 = Matrix(arm.jnts[1].loc_pos)
p23 = Matrix(arm.jnts[2].loc_pos)
p34 = Matrix(arm.jnts[3].loc_pos)
p45 = Matrix(arm.jnts[4].loc_pos)
p56 = Matrix(arm.jnts[5].loc_pos)


def stheta(x):
    return 2 * x / (x ** 2 + 1)


def ctheta(x):
    return (x ** 2 - 1) / (x ** 2 + 1)

def hat(v):
    return Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


R01 = k0 * k0.T + stheta(x0) * hat(k0) + ctheta(x0) * hat(k0) * hat(k0)
R12 = k1 * k1.T + stheta(x1) * hat(k1) + ctheta(x1) * hat(k1) * hat(k1)
R23 = k2 * k2.T + stheta(x2) * hat(k2) + ctheta(x2) * hat(k2) * hat(k2)
R34 = k3 * k3.T + stheta(x3) * hat(k3) + ctheta(x3) * hat(k3) * hat(k3)
R45 = k4 * k4.T + stheta(x4) * hat(k4) + ctheta(x4) * hat(k4) * hat(k4)
R56 = k5 * k5.T + stheta(x5) * hat(k5) + ctheta(x5) * hat(k5) * hat(k5)

# tgt_rotmat = R01 @ R12 @ R23 @ R34 @ R45 @ R56
# tgt_pos = p01 + R01 @ p12 + R01 @ R12 @ p23 + R01 @ R12 @ R23 @ p34 + R01 @ R12 @ R23 @ R34 @ p45 + R01 @ R12 @ R23 @ R34 @ R45 @ p56

pprint(solve(k0 * k0.T + stheta(x0) * hat(k0) + ctheta(x0) * hat(k0) * hat(k0)-tR, x0))
