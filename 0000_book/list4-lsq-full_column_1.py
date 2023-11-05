import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.constant as cnst

if __name__ == '__main__':
    base = wd.World(cam_pos=np.array([.15, .1, 1]), lookat_pos=np.array([.15, .1, 0]),
                    lens_type=wd.LensType.PERSPECTIVE)
    gm.gen_arrow(spos=np.zeros(3), epos=np.array([.3, 0, 0]), rgba=cnst.black, stick_type="round").attach_to(base)
    gm.gen_arrow(spos=np.zeros(3), epos=np.array([0, .2, 0]), rgba=cnst.black, stick_type="round").attach_to(base)
    A = np.array([[-1, 2],
                  [.2, 1],
                  [.4, .2]])
    b = np.array([0.1, 0.1, 0.1])
    spos = np.zeros((3, 3))
    spos[:, 0] = 1
    spos[:, 1] = np.divide(b - A[:, 0], A[:, 1])
    epos = np.zeros((3, 3))
    epos[:, 0] = -1
    epos[:, 1] = np.divide(b + A[:, 0], A[:, 1])

    gm.gen_stick(spos[0, :], epos[0, :], rgba=cnst.magenta).attach_to(base)
    gm.gen_stick(spos[1, :], epos[1, :], rgba=cnst.yellow).attach_to(base)
    gm.gen_stick(spos[2, :], epos[2, :], rgba=cnst.cyan).attach_to(base)

    # A_norm = np.linalg.norm(A, axis=1)
    # A_tf = np.linalg.inv(np.diag(A_norm))
    # A = A_tf @ A
    # b = A_tf @ b
    # lsq_x = np.linalg.inv(A.T @ A) @ A.T @ b
    lsq_x = np.linalg.pinv(A) @ b
    lsq_pos = np.zeros(3)
    lsq_pos[:2] = lsq_x
    gm.gen_sphere(lsq_pos, radius=.0075, rgba=cnst.oriental_blue).attach_to(base)
    base.run()
