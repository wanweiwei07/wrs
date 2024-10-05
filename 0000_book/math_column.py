from wrs import wd, rm, mgm

base = wd.World(cam_pos=rm.np.array([1, 1, 1]), lookat_pos=rm.np.array([0, 0, 0]), toggle_debug=True)
mgm.gen_frame(ax_length=.2).attach_to(base)

base.run()