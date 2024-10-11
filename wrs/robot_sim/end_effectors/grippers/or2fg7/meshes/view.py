from wrs import wd, rm, mgm

file_name = "inward_left_finger_link.stl"

base = wd.World(cam_pos=rm.vec(.5, .5, .5), lookat_pos=rm.vec(0, 0, 0), auto_cam_rotate=False)
obj = mgm.GeometricModel(initor=file_name)
obj.attach_to(base)
mgm.gen_frame().attach_to(obj)

base.run()
