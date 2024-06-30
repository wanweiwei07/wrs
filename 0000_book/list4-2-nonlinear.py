import time
import numpy as np
import modeling.collision_model as cm
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.constant as cnst
import basis.robot_math as rm

def nonlinear_function(x):
    return 0.05*np.sin(40*x)+0.05*np.cos(20*x)+0.5*x**2+0.1*x+.08

base = wd.World(cam_pos=np.array([.15, .1, 1]), lookat_pos=np.array([.15, .1, 0]),
                lens_type=wd.LensType.PERSPECTIVE)
gm.gen_arrow(spos=np.zeros(3), epos=np.array([.3,0,0]), rgb=rm.bc.black, alpha=1, stick_type="round").attach_to(base)
gm.gen_arrow(spos=np.zeros(3), epos=np.array([0,.2,0]), rgb=rm.bc.black, alpha=1, stick_type="round").attach_to(base)

x_values = np.linspace(0,0.3,400)
y = nonlinear_function(x_values)
for i in range(len(x_values)-1):
    gm.gen_stick(spos=np.array([x_values[i], y[i], 0]), epos=np.array([x_values[i+1], y[i+1], 0]),
                 rgb=rm.bc.orange).attach_to(base)
base.run()
#
A = np.array([[-1, 2],
              [.2, 1],])
b = np.array([0.1, 0.1])
spos = np.zeros((2, 3))
spos[:, 0] = 1
spos[:, 1] = np.divide(b - A[:, 0], A[:, 1])
epos = np.zeros((2, 3))
epos[:, 0] = -1
epos[:, 1] = np.divide(b + A[:, 0], A[:, 1])

gm.gen_stick(spos[0, :], epos[0, :], rgba=cnst.magenta).attach_to(base)
gm.gen_stick(spos[1, :], epos[1, :], rgba=cnst.yellow).attach_to(base)

# lsq_x = np.linalg.inv(A) @ b
# lsq_pos = np.zeros(3)
# lsq_pos[0:2] = lsq_x
# mgm.gen_sphere(lsq_pos, radius=.01, rgba=bc.deep_sky_blue).attach_to(base)

base.run()