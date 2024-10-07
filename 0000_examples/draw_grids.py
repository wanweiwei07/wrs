from wrs import wd, rm, mgm

base = wd.World(cam_pos=[1, .7, .3], lookat_pos=[0, 0, 0])
for x in [-.03, 0, .03]:
    for y in [-.03, 0, .03]:
        for z in [-.03, 0, .03]:
            mgm.gen_frame_box(xyz_lengths=[.03, .03, .03], pos=rm.vec(x, y, z), rotmat=rm.np.eye(3),
                              rgb=rm.const.black, thickness=.00001).attach_to(base)
for x in [0]:
    for y in [0]:
        for z in [-.03, 0, .03]:
            homomat = rm.np.eye(4)
            homomat[:3, 3] = rm.np.array([x, y, z])
            mgm.gen_box(xyz_lengths=[.03, .03, .03], pos=rm.vec(x, y, z), rotmat=rm.np.eye(3),
                        rgb=rm.const.yellow).attach_to(base)
mgm.gen_box(xyz_lengths=[.03, .03, .03], pos=rm.vec(.03, 0, -.03),
            rotmat=rm.np.eye(3), rgb=rm.const.yellow).attach_to(base)
base.run()
