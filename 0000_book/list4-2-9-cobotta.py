import time
from wrs import wd, rm, mgm, cbt

base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
mgm.gen_frame().attach_to(base)
robot = cbt.Cobotta(name="wrs_cobotta", enable_cc=True)
robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
tgt_pos = rm.np.array([.3, .1, .3])
tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], rm.pi * 2 / 3)
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
tic = time.time()
jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
toc = time.time()
print("ik cost: ", toc - tic)
print(jnt_values)
if jnt_values is not None:
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
base.run()