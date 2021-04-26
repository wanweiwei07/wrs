import visualization.panda.world as wd
import modeling.geometric_model as gm
import basis.robot_math as rm
import math

base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0], toggle_debug=True)
frame_o = gm.gen_frame(length=.2)
frame_o.attach_to(base)
# rotmat = rm.rotmat_from_axangle([1,1,1],math.pi/4)
rotmat = rm.rotmat_from_euler(math.pi/3, -math.pi/6, math.pi/3)
# frame_a = gm.gen_mycframe(length=.2, rotmat=rotmat)
frame_a = gm.gen_dashframe(length=.2, rotmat=rotmat)
frame_a.attach_to(base)

print(rotmat)
base.run()