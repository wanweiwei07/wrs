import time
from wrs import basis as rm, robot_sim as rs007l, modeling as mgm
import wrs.visualization.panda.world as wd

base = wd.World(cam_pos=[5, 0, 3], lookat_pos=[0, 0, .7])
mgm.gen_frame().attach_to(base)
robot = rs007l.RS007L(enable_cc=True)
# robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
# robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
tgt_pos = rm.np.array([.35, .3, 1])
tgt_rotmat = rm.rotmat_from_euler(rm.np.radians(130), rm.np.radians(40), rm.np.radians(180))
mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
tic = time.perf_counter()
jnt_values = robot.ik(tgt_pos=tgt_pos,
                  tgt_rotmat=tgt_rotmat)
toc = time.perf_counter()
print("ik cost: ", toc - tic)
print(jnt_values)
if jnt_values is not None:
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
base.run()