from wrs import wd, rm, mgm
import wrs.robot_sim.manipulators.irb14050.irb14050 as irb

base = wd.World(cam_pos=rm.vec(1.5, 1, 0.7), lookat_pos=rm.vec(0, 0, .2))
mgm.gen_frame().attach_to(base)
arm = irb.IRB14050(enable_cc=True)
arm.gen_stickmodel(toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
arm.gen_meshmodel(toggle_cdprim=True, alpha=1).attach_to(base)
print(arm.is_collided())
base.run()
