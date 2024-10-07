from wrs import wd, mgm

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.vec(1, 1, 1), lookat_pos=rm.np.zeros(3))
    frame_model = mgm.gen_frame(ax_length=.2)
    frame_model.attach_to(base)
    base.run()