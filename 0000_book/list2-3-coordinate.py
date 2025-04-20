from wrs import wd, rm, mgm

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.vec(1, .8, .6), lookat_pos=rm.zeros(3))
    frame_model = mgm.gen_frame()
    frame_model.attach_to(base)
    base.run()