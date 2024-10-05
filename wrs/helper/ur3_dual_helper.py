import numpy as np
import wrs.robot_con.ur.ur3_dual_x as ur3dx
import wrs.motion.optimization_based.incremental_nik as inik
import wrs.visualization.panda.world as wd
from wrs import robot_sim as ur3ds, manipulation as ppp, motion as rrtc

class UR3DualHelper(object):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 use_real=False,
                 create_sim_world=True,
                 lft_robot_ip='10.2.0.50',
                 rgt_robot_ip='10.2.0.51',
                 pc_ip='10.2.0.100',
                 cam_pos=np.array([2, 1, 3]),
                 lookat_pos=np.array([0, 0, 1.1]),
                 auto_cam_rotate=False):
        self.robot_s = ur3ds.UR3Dual(pos=pos, rotmat=rotmat)
        self.rrt_planner = rrtc.RRTConnect(self.robot_s)
        self.inik_solver = inik.IncrementalNIK(self.robot_s)
        self.pp_planner = ppp.PickPlacePlanner(self.robot_s)
        if use_real:
            self.robot_x = ur3dx.UR3DualX(lft_robot_ip=lft_robot_ip,
                                          rgt_robot_ip=rgt_robot_ip,
                                          pc_ip=pc_ip)
        if create_sim_world:
            self.sim_world = wd.World(cam_pos=cam_pos,
                                      lookat_pos=lookat_pos,
                                      auto_cam_rotate=auto_cam_rotate)

    def plan_motion(self,
                    component_name,
                    start_conf,
                    goal_conf,
                    obstacle_list=[],
                    otherrobot_list=[],
                    ext_dist=2,
                    maxiter=1000,
                    maxtime=15.0,
                    animation=False):
        path = self.rrt_planner.plan(component_name=component_name,
                                     start_conf=start_conf,
                                     goal_conf=goal_conf,
                                     obstacle_list=obstacle_list,
                                     other_robot_list=otherrobot_list,
                                     ext_dist=ext_dist,
                                     max_iter=maxiter,
                                     max_time=maxtime,
                                     animation=animation)
        return path

    def plan_pick_and_place(self,
                            manipulator_name,
                            hand_name,
                            objcm,
                            grasp_info_list,
                            start_conf,
                            goal_homomat_list):
        """
        :param manipulator_name:
        :param hand_name:
        :param objcm:
        :param grasp_info_list:
        :param start_conf:
        :param goal_homomat_list:
        :return:
        author: weiwei
        date: 20210409
        """
        self.pp_planner.gen_pick_and_place_motion(manipulator_name,
                                                  hand_name,
                                                  objcm,
                                                  grasp_info_list,
                                                  start_conf,
                                                  goal_homomat_list)