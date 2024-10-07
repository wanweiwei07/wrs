import time
from wrs import wd, rm, mgm, ko2fg

base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, .7])
mgm.gen_frame().attach_to(base)
robot = ko2fg.KHI_OR2FG7(enable_cc=True)
robot.gen_meshmodel(alpha=.3, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
# base.run()
tgt_pos = rm.vec(.35, .3, 1)
tgt_rotmat = rm.rotmat_from_euler(rm.radians(130), rm.radians(40), rm.radians(180))
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
tic = time.perf_counter()
jnt_values = robot.ik(tgt_pos=tgt_pos,
                      tgt_rotmat=tgt_rotmat)
toc = time.perf_counter()
print("ik cost: ", toc - tic)
print(jnt_values)
if jnt_values is not None:
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
base.run()
