from wrs import wd, rm, mgm, rrtc, cbt

base = wd.World(cam_pos=[1.5, 1.5, .75], lookat_pos=[0, 0, .2])
mgm.gen_frame().attach_to(base)

robot = cbt.Cobotta(enable_cc=True)
start_conf = robot.get_jnt_values()
robot.goto_given_conf(jnt_values=start_conf)
robot.gen_meshmodel(rgb=rm.const.winter_map(0.0), alpha=1).attach_to(base)

goal_conf = robot.rand_conf()
robot.goto_given_conf(jnt_values=goal_conf)
robot.gen_meshmodel(rgb=rm.const.winter_map(1.0), alpha=1).attach_to(base)

rrtc_planner = rrtc.RRTConnect(robot)
mot_data = rrtc_planner.plan(start_conf=start_conf,
                             goal_conf=goal_conf,
                             ext_dist=.1,
                             max_time=300)
if mot_data is not None:
    n_step = len(mot_data.mesh_list)
    for i, model in enumerate(mot_data.mesh_list):
        model.rgb = rm.const.winter_map(i / n_step)
        model.alpha = .3
        model.attach_to(base)
else:
    print("No available motion found.")

import wrs.motion.trajectory.piecewisepoly_toppra as topp
# import motion.trajectory.topp as topp

traj_planner= topp.PiecewisePolyTOPPRA()
new_path = traj_planner.interpolate_by_max_spdacc(path=mot_data.jv_list,
                                                  max_vels=[rm.pi/2]*6,
                                                  max_accs=[rm.pi/2]*6)
for i, jnt_values in enumerate(new_path):
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel(rgb=rm.const.summer_map(i / len(new_path)), alpha=.3).attach_to(base)
base.run()
