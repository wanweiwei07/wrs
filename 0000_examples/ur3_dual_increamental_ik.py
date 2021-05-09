import motion.optimization_based.incremental_nik as inik
import visualization.panda.world as wd
import modeling.geometric_model as gm
import modeling.collision_model as cm
import robot_sim.robots.ur3_dual.ur3_dual as ur3d
import numpy as np
import math
import basis.robot_math as rm

if __name__ == '__main__':
    base = wd.World(cam_pos=[2, 1, 3], lookat_pos=[0, 0, 1.1])
    gm.gen_frame().attach_to(base)
    # object
    object = cm.CollisionModel("./objects/bunnysim.stl")
    object.set_pos(np.array([.55, -.3, 1.3]))
    object.set_rgba([.5, .7, .3, 1])
    object.attach_to(base)
    # robot_s
    component_name = 'rgt_arm'
    robot_instance = ur3d.UR3Dual()
    start_hnd_pos=np.array([0.4, -0.5, 1.3])
    start_hnd_rotmat=rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_hnd_pos=np.array([0.4, -0.3, 1.3])
    goal_hnd_rotmat=rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=start_hnd_pos, rotmat=start_hnd_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_hnd_pos, rotmat=goal_hnd_rotmat).attach_to(base)

    robot_inik_solver = inik.IncrementalNIK(robot_instance)
    pose_list = robot_inik_solver.gen_linear_motion(component_name,
                                                    start_tcp_pos=start_hnd_pos,
                                                    start_tcp_rotmat=start_hnd_rotmat,
                                                    goal_tcp_pos=goal_hnd_pos,
                                                    goal_tcp_rotmat=goal_hnd_rotmat,
                                                    obstacle_list=[object])

    for jnt_values in pose_list:
        robot_instance.fk(component_name, jnt_values)
        robot_meshmodel = robot_instance.gen_meshmodel()
        robot_meshmodel.attach_to(base)
    base.run()