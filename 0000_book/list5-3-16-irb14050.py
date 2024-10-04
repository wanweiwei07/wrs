import wrs.visualization.panda.world as wd
from wrs import robot_sim as irb, modeling as mgm

base = wd.World(cam_pos=[1.5, 1, 0.7], lookat_pos=[0, 0, .2])
mgm.gen_frame().attach_to(base)
arm = irb.IRB14050(enable_cc=True)
arm.gen_stickmodel(toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
arm.gen_meshmodel(toggle_cdprim=True, alpha=1).attach_to(base)
print(arm.is_collided())
base.run()