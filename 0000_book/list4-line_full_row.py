import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as cnst, modeling as gm

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.15, .1, 1]), lookat_pos=np.array([.15, .1, 0]),
                    lens_type=wd.LensType.PERSPECTIVE)
    gm.gen_arrow(spos=np.zeros(3), epos=np.array([.3, 0, 0]), rgba=cnst.black, stick_type="round").attach_to(base)
    gm.gen_arrow(spos=np.zeros(3), epos=np.array([0, .2, 0]), rgba=cnst.black, stick_type="round").attach_to(base)

    A = np.array([[-1, 2]])
    b = np.array([0.1])
    spos = np.zeros((1, 3))
    spos[:, 0] = 1
    spos[:, 1] = np.divide(b - A[:, 0], A[:, 1])
    epos = np.zeros((1, 3))
    epos[:, 0] = -1
    epos[:, 1] = np.divide(b + A[:, 0], A[:, 1])

    gm.gen_stick(spos[0, :], epos[0, :], rgba=cnst.magenta).attach_to(base)

    lsq_x = A.T @ np.linalg.inv(A @ A.T) @ b
    lsq_pos = np.zeros(3)
    lsq_pos[:2] = lsq_x
    gm.gen_sphere(lsq_pos, radius=.0075, rgba=cnst.oriental_blue).attach_to(base)
    base.run()
