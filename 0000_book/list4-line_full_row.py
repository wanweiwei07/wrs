from wrs import wd, rm, mgm

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.np.array([.15, .1, 1]), lookat_pos=rm.np.array([.15, .1, 0]),
                    lens_type=wd.LensType.PERSPECTIVE)
    mgm.gen_arrow(spos=rm.np.zeros(3), epos=rm.np.array([.3, 0, 0]), rgb=rm.const.black, stick_type="round").attach_to(
        base)
    mgm.gen_arrow(spos=rm.np.zeros(3), epos=rm.np.array([0, .2, 0]), rgb=rm.const.black, stick_type="round").attach_to(
        base)

    A = rm.np.array([[-1, 2]])
    b = rm.np.array([0.1])
    spos = rm.np.zeros((1, 3))
    spos[:, 0] = 1
    spos[:, 1] = rm.np.divide(b - A[:, 0], A[:, 1])
    epos = rm.np.zeros((1, 3))
    epos[:, 0] = -1
    epos[:, 1] = rm.np.divide(b + A[:, 0], A[:, 1])

    mgm.gen_stick(spos[0, :], epos[0, :], rgb=rm.const.magenta).attach_to(base)

    lsq_x = A.T @ rm.np.linalg.inv(A @ A.T) @ b
    lsq_pos = rm.np.zeros(3)
    lsq_pos[:2] = lsq_x
    mgm.gen_sphere(lsq_pos, radius=.0075, rgb=rm.const.oriental_blue).attach_to(base)
    base.run()
