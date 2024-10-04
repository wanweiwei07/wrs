import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.motion.probabilistic.rrt_connect as rrtc
import wrs.motion.primitives.interpolated as mpi


class ADPlanner(object):
    """
    AD = Approach_Depart
    NOTE: only accept robot as initiator
    """

    def __init__(self, robot):
        """
        :param robot_s:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        if not isinstance(robot, sari.SglArmRobotInterface):
            if not hasattr(robot, "delegator") or not isinstance(robot.delegator, sari.SglArmRobotInterface):
                raise ValueError("Only single arm robot can be used to initiate an InterplateMotion instance!")
        self.robot = robot
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.im_planner = mpi.InterplatedMotion(self.robot)

    def gen_linear_approach(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            direction=None,
                            distance=.07,
                            ee_values=None,
                            granularity=0.02,
                            obstacle_list=None):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param direction: use the loc_z of goal_tcp_rotmat if None
        :param distance:
        :param ee_values: values for end-effector
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values
        :return:
        author: weiwei
        date: 20210125
        """
        mot_data = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=goal_tcp_pos,
                                                         goal_tcp_rotmat=goal_tcp_rotmat,
                                                         direction=direction,
                                                         distance=distance,
                                                         obstacle_list=obstacle_list,
                                                         granularity=granularity,
                                                         type="sink",
                                                         ee_values=ee_values)
        if mot_data is None:
            print("ADPlanner: Cannot generate linear approach!")
            return None
        else:
            return mot_data

    def gen_linear_depart(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          direction=None,  # np.array([0, 0, 1])
                          distance=.07,
                          ee_values=None,
                          granularity=0.03,
                          obstacle_list=None):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param direction:
        :param distance:
        :param ee_values:
        :param granularity:
        :param seed_jnt_values:
        :return:
        author: weiwei
        date: 20210125
        """
        mot_data = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=start_tcp_pos,
                                                         goal_tcp_rotmat=start_tcp_rotmat,
                                                         direction=direction,
                                                         distance=distance,
                                                         obstacle_list=obstacle_list,
                                                         granularity=granularity,
                                                         type='source',
                                                         ee_values=ee_values)
        if mot_data is None:
            print("ADPlanner: Cannot generate linear depart!")
            return None
        else:
            return mot_data

    def gen_linear_approach_depart(self,
                                   goal_tcp_pos,
                                   goal_tcp_rotmat,
                                   approach_direction=None,  # np.array([0, 0, -1])
                                   approach_distance=.07,
                                   approach_ee_values=None,
                                   depart_direction=None,  # np.array([0, 0, 1])
                                   depart_distance=.07,
                                   depart_ee_values=None,
                                   granularity=.02,
                                   obstacle_list=None):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param hnd_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_ee_values:
        :param depart_direction:
        :param depart_distance:
        :param depart_ee_values:
        :param granularity:
        :param obstacle_list:
        :return:
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        app_mot_data = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=goal_tcp_pos,
                                                             goal_tcp_rotmat=goal_tcp_rotmat,
                                                             direction=approach_direction,
                                                             distance=approach_distance,
                                                             obstacle_list=obstacle_list,
                                                             granularity=granularity,
                                                             type="sink",
                                                             ee_values=approach_ee_values)
        if app_mot_data is None:
            print("ADPlanner: Cannot generate the approach section of linear approach-depart!")
            return None
        else:
            dep_mot_data = self.im_planner.gen_rel_linear_motion(goal_tcp_pos=goal_tcp_pos,
                                                                 goal_tcp_rotmat=goal_tcp_rotmat,
                                                                 direction=depart_direction,
                                                                 distance=depart_distance,
                                                                 obstacle_list=obstacle_list,
                                                                 granularity=granularity,
                                                                 type='source',
                                                                 ee_values=depart_ee_values)
            if dep_mot_data is None:
                print("ADPlanner: Cannot generate the depart section of linear approach-depart!")
                return None
            else:
                return app_mot_data + dep_mot_data

    def gen_linear_approach_to_given_conf(self,
                                          goal_jnt_values,
                                          direction=None,
                                          distance=.07,
                                          ee_values=None,
                                          granularity=0.02,
                                          obstacle_list=None):
        mot_data = self.im_planner.gen_rel_linear_motion_with_given_conf(goal_jnt_values=goal_jnt_values,
                                                                         direction=direction,
                                                                         distance=distance,
                                                                         obstacle_list=obstacle_list,
                                                                         granularity=granularity,
                                                                         type="sink",
                                                                         ee_values=ee_values)
        if mot_data is None:
            print('ADPlanner: Cannot generate linear approach with given jnt values!')
            return None
        else:
            return mot_data

    def gen_linear_depart_from_given_conf(self,
                                          start_jnt_values,
                                          direction=None,
                                          distance=.07,
                                          ee_values=None,
                                          granularity=0.02,
                                          obstacle_list=None):
        mot_data = self.im_planner.gen_rel_linear_motion_with_given_conf(goal_jnt_values=start_jnt_values,
                                                                         direction=direction,
                                                                         distance=distance,
                                                                         obstacle_list=obstacle_list,
                                                                         granularity=granularity,
                                                                         type="source",
                                                                         ee_values=ee_values)
        if mot_data is None:
            print("ADPlanner: Cannot generate linear approach with given jnt values!")
            return None
        else:
            return mot_data

    def gen_linear_approach_depart_with_given_conf(self,
                                                   goal_jnt_values,
                                                   approach_direction=None,  # np.array([0, 0, -1])
                                                   approach_distance=.07,
                                                   approach_ee_values=None,
                                                   depart_direction=None,  # np.array([0, 0, 1])
                                                   depart_distance=.07,
                                                   depart_ee_values=None,
                                                   granularity=.02,
                                                   obstacle_list=None):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param approach_direction:
        :param approach_distance:
        :param approach_ee_values:
        :param depart_direction:
        :param depart_distance:
        :param depart_ee_values:
        :param granularity:
        :param obstacle_list:
        :return:
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        app_mot_data = self.gen_linear_approach_to_given_conf(goal_jnt_values=goal_jnt_values,
                                                              direction=approach_direction,
                                                              distance=approach_distance,
                                                              ee_values=approach_ee_values,
                                                              granularity=granularity,
                                                              obstacle_list=obstacle_list)
        if app_mot_data is None:
            print("ADPlanner: Cannot generate the approach section of linear approach-depart with given conf!")
            return None
        else:
            dep_mot_data = self.gen_linear_depart_from_given_conf(start_jnt_values=goal_jnt_values,
                                                                  direction=depart_direction,
                                                                  distance=depart_distance,
                                                                  ee_values=depart_ee_values,
                                                                  granularity=granularity,
                                                                  obstacle_list=obstacle_list)
            if dep_mot_data is None:
                print("ADPlanner: Cannot generate the depart section of linear approach-depart with given conf!")
                return None
            else:
                return app_mot_data + dep_mot_data

    def gen_approach(self,
                     goal_tcp_pos,
                     goal_tcp_rotmat,
                     start_jnt_values=None,
                     linear_direction=None,
                     linear_distance=.07,
                     ee_values=None,
                     granularity=.02,
                     obstacle_list=None,  #
                     object_list=None,  #
                     use_rrt=True):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_jnt_values:
        :param linear_direction:
        :param linear_distance:
        :param ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :param use_rrt:
        :return:
        """
        if obstacle_list is None:
            obstacle_list = []
        if object_list is None:
            object_list = []
        linear_app = self.gen_linear_approach(goal_tcp_pos=goal_tcp_pos,
                                              goal_tcp_rotmat=goal_tcp_rotmat,
                                              direction=linear_direction,
                                              distance=linear_distance,
                                              ee_values=ee_values,  # do not change jaw width
                                              granularity=granularity,
                                              obstacle_list=obstacle_list)
        if linear_app is None:
            print("ADPlanner: Cannot generate the linear approach section of gen_approach!")
            return None
        if start_jnt_values is None:
            return linear_app
        if use_rrt:
            ee_values_bk = None
            if ee_values is not None:
                ee_values_bk = self.robot.get_ee_values()
                self.robot.change_ee_values(ee_values=ee_values)
            start2app = self.rrtc_planner.plan(start_conf=start_jnt_values,
                                               goal_conf=linear_app.jv_list[0],
                                               obstacle_list=obstacle_list + object_list,
                                               ext_dist=.1,
                                               max_time=100)
            if ee_values is not None:
                self.robot.change_ee_values(ee_values=ee_values_bk)
            if start2app is None:
                print("ADPlanner: Cannot plan the rrt motion from start_jnt_values to the beginning of approach!")
                return None
        else:
            start2app = self.im_planner.gen_interplated_between_given_conf(start_jnt_values=start_jnt_values,
                                                                           end_jnt_values=linear_app.jv_list[0],
                                                                           obstacle_list=obstacle_list + object_list,
                                                                           ee_values=ee_values)
            if start2app is None:
                print("ADPlanner: Cannot interpolate the motion from start_jnt_values to the beginning of approach!")
                return None
        return start2app + linear_app

    def gen_depart(self,
                   start_tcp_pos,
                   start_tcp_rotmat,
                   end_jnt_values=None,
                   linear_direction=None,
                   linear_distance=.07,
                   ee_values=None,
                   granularity=.02,
                   obstacle_list=None,  #
                   object_list=None,
                   use_rrt=True):  #
        """
        :param start_tcp_pos:
        :param start_tcp_rotmat:
        :param end_jnt_values:
        :param linear_direction:
        :param linear_distance:
        :param ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :param use_rrt:
        :return:
        """
        if obstacle_list is None:
            obstacle_list = []
        if object_list is None:
            object_list = []
        linear_dep = self.gen_linear_depart(start_tcp_pos=start_tcp_pos,
                                            start_tcp_rotmat=start_tcp_rotmat,
                                            direction=linear_direction,
                                            distance=linear_distance,
                                            ee_values=ee_values,
                                            granularity=granularity,
                                            obstacle_list=obstacle_list)
        if linear_dep is None:
            print("ADPlanner: Cannot gen depart linear!")
            return None
        if end_jnt_values is None:
            return linear_dep
        if use_rrt:
            ee_values_bk = None
            if ee_values is not None:
                ee_values_bk = self.robot.get_ee_values()
                self.robot.change_ee_values(ee_values=ee_values)
            dep2end = self.rrtc_planner.plan(start_conf=linear_dep.jv_list[-1],
                                             goal_conf=end_jnt_values,
                                             obstacle_list=obstacle_list + object_list,
                                             ext_dist=.1,
                                             max_time=100)
            if ee_values is not None:
                self.robot.change_ee_values(ee_values=ee_values_bk)
        else:
            dep2end = self.im_planner.gen_interplated_between_given_conf(start_jnt_values=linear_dep.jv_list[-1],
                                                                         end_jnt_values=end_jnt_values,
                                                                         obstacle_list=obstacle_list + object_list,
                                                                         ee_values=ee_values)
        if dep2end is None:
            print("ADPlanner: Cannot plan depart motion!")
            return None
        return linear_dep + dep2end

    def gen_approach_depart(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            start_jnt_values=None,
                            end_jnt_values=None,
                            approach_direction=None,
                            approach_distance=.1,
                            approach_ee_values=None,
                            depart_direction=None,
                            depart_distance=.1,
                            depart_ee_values=None,
                            granularity=.03,
                            obstacle_list=None,
                            object_list=None,
                            use_rrt=True):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_jnt_values:
        :param end_jnt_values:
        :param approach_direction: last column of rotmat by default
        :param approach_distance:
        :param approach_ee_values:
        :param depart_direction:
        :param depart_distance:
        :param depart_ee_values:
        :param granularity: obstacles, will be checked by both rrt and linear
        :param obstacle_list: target objects, will be checked by rrt, but not by linear
        :param object_list:
        :param use_rrt
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        app_mot_data = self.gen_approach(goal_tcp_pos=goal_tcp_pos,
                                         goal_tcp_rotmat=goal_tcp_rotmat,
                                         start_jnt_values=start_jnt_values,
                                         linear_direction=approach_direction,
                                         linear_distance=approach_distance,
                                         ee_values=approach_ee_values,
                                         granularity=granularity,
                                         obstacle_list=obstacle_list,
                                         object_list=object_list,
                                         use_rrt=use_rrt)
        if app_mot_data is None:
            print("ADPlanner: Cannot plan the approach section of approach-depart motion!")
            return None
        else:
            dep_mot_data = self.gen_depart_from_given_conf(start_jnt_values=app_mot_data.jv_list[-1],
                                                           end_jnt_values=end_jnt_values,
                                                           linear_direction=depart_direction,
                                                           linear_distance=depart_distance,
                                                           ee_values=depart_ee_values,
                                                           granularity=granularity,
                                                           obstacle_list=obstacle_list,
                                                           object_list=object_list,
                                                           use_rrt=use_rrt)
            if dep_mot_data is None:
                print("ADPlanner: Cannot plan the depart section of approach-depart motion!")
                return None
            else:
                return app_mot_data + dep_mot_data

    def gen_depart_approach_with_given_conf(self,
                                            start_jnt_values=None,
                                            end_jnt_values=None,
                                            depart_direction=None,
                                            depart_distance=.1,
                                            depart_ee_values=None,
                                            approach_direction=None,
                                            approach_distance=.1,
                                            approach_ee_values=None,
                                            granularity=.03,
                                            obstacle_list=None,
                                            object_list=None,
                                            use_rrt=True):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_jnt_values:
        :param end_jnt_values:
        :param approach_direction:
        :param approach_distance:
        :param approach_ee_values:
        :param depart_direction:
        :param depart_distance:
        :param depart_ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :param use_rrt
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        linear_dep_mot_data = self.gen_linear_depart_from_given_conf(start_jnt_values=start_jnt_values,
                                                                     direction=depart_direction,
                                                                     distance=depart_distance,
                                                                     ee_values=depart_ee_values,
                                                                     granularity=granularity,
                                                                     obstacle_list=obstacle_list)
        if linear_dep_mot_data is None:
            print("ADPlanner: Cannot plan the linear depart section of depart-approach motion with given conf!")
            return None
        else:
            app_mot_data = self.gen_approach_to_given_conf(goal_jnt_values=end_jnt_values,
                                                           start_jnt_values=linear_dep_mot_data.jv_list[-1],
                                                           linear_direction=approach_direction,
                                                           linear_distance=approach_distance,
                                                           ee_values=approach_ee_values,
                                                           granularity=granularity,
                                                           obstacle_list=obstacle_list,
                                                           object_list=object_list,
                                                           use_rrt=use_rrt)
            if app_mot_data is None:
                print("ADPlanner: Cannot plan the approach section of depart-approach motion given conf!")
                return None
            else:
                return linear_dep_mot_data + app_mot_data

    def gen_approach_to_given_conf(self,
                                   goal_jnt_values,
                                   start_jnt_values=None,
                                   linear_direction=None,
                                   linear_distance=.1,
                                   ee_values=None,
                                   granularity=.03,
                                   obstacle_list=None,
                                   object_list=None,
                                   use_rrt=True):
        """
        :param goal_jnt_values:
        :param start_jnt_values:
        :param linear_direction:
        :param linear_distance:
        :param ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :param use_rrt
        :return:
        """
        if obstacle_list is None:
            obstacle_list = []
        if object_list is None:
            object_list = []
        linear_app = self.gen_linear_approach_to_given_conf(goal_jnt_values=goal_jnt_values,
                                                            direction=linear_direction,
                                                            distance=linear_distance,
                                                            ee_values=ee_values,
                                                            granularity=granularity,
                                                            obstacle_list=obstacle_list)
        if linear_app is None:
            print("ADPlanner: Cannot plan the linear approach section of approach with given conf!")
            return None
        if start_jnt_values is None:
            return linear_app
        if use_rrt:
            ee_values_bk = None
            if ee_values is not None:
                ee_values_bk = self.robot.get_ee_values()
                self.robot.change_ee_values(ee_values=ee_values)
            start2app = self.rrtc_planner.plan(start_conf=start_jnt_values,
                                               goal_conf=linear_app.jv_list[0],
                                               obstacle_list=obstacle_list + object_list,
                                               ext_dist=.1,
                                               max_time=100)
            if ee_values is not None:
                self.robot.change_ee_values(ee_values=ee_values_bk)
        else:
            start2app = self.im_planner.gen_interplated_between_given_conf(start_jnt_values=start_jnt_values,
                                                                           end_jnt_values=linear_app.jv_list[0],
                                                                           obstacle_list=obstacle_list + object_list,
                                                                           ee_values=ee_values)
        if start2app is None:
            print("ADPlanner: Cannot plan the approach rrt motion section of approach with given conf!")
            return None
        return start2app + linear_app

    def gen_depart_from_given_conf(self,
                                   start_jnt_values,
                                   end_jnt_values=None,
                                   linear_direction=None,
                                   linear_distance=.1,
                                   ee_values=None,
                                   granularity=.03,
                                   obstacle_list=None,
                                   object_list=None,
                                   use_rrt=True):
        """
        :param start_jnt_values:
        :param end_jnt_values:
        :param linear_direction:
        :param linear_distance:
        :param ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :param use_rrt
        :return:
        """
        if obstacle_list is None:
            obstacle_list = []
        if object_list is None:
            object_list = []
        linear_dep = self.gen_linear_depart_from_given_conf(start_jnt_values=start_jnt_values,
                                                            direction=linear_direction,
                                                            distance=linear_distance,
                                                            ee_values=ee_values,
                                                            granularity=granularity,
                                                            obstacle_list=obstacle_list)
        if linear_dep is None:
            print("ADPlanner: Cannot plan the linear depart section of depart with given conf!")
            return None
        if end_jnt_values is None:
            return linear_dep
        if use_rrt:
            ee_values_bk = None
            if ee_values is not None:
                ee_values_bk = self.robot.get_ee_values()
                self.robot.change_ee_values(ee_values=ee_values)
            dep2end = self.rrtc_planner.plan(start_conf=linear_dep.jv_list[-1],
                                             goal_conf=end_jnt_values,
                                             obstacle_list=obstacle_list + object_list,
                                             ext_dist=.1,
                                             max_time=100)
            if ee_values is not None:
                self.robot.change_ee_values(ee_values=ee_values_bk)
            if dep2end is None:
                print("ADPlanner: Cannot plan the depart rrt motion section of depart with given conf!")
                return None
        else:
            dep2end = self.im_planner.gen_interplated_between_given_conf(start_jnt_values=linear_dep.jv_list[-1],
                                                                         end_jnt_values=end_jnt_values,
                                                                         obstacle_list=obstacle_list + object_list,
                                                                         ee_values=ee_values)
            if dep2end is None:
                print("ADPlanner: Cannot interpolate the depart motion section of depart with given conf!")
                return None
        return linear_dep + dep2end

    def gen_approach_depart_with_given_conf(self,
                                            goal_jnt_values,
                                            start_jnt_values=None,
                                            end_jnt_values=None,
                                            approach_direction=None,  # np.array([0, 0, -1])
                                            approach_distance=.1,
                                            approach_ee_values=None,
                                            depart_direction=None,  # np.array([0, 0, 1])
                                            depart_distance=.1,
                                            depart_ee_values=None,
                                            granularity=.03,
                                            obstacle_list=None,  #
                                            object_list=None,
                                            use_rrt=True):  #
        """
        :param goal_jnt_values
        :param start_jnt_values:
        :param end_jnt_values:
        :param approach_direction:
        :param approach_distance:
        :param approach_ee_values:
        :param depart_direction:
        :param depart_distance:
        :param depart_ee_values:
        :param granularity:
        :param obstacle_list: obstacles, will be checked by both rrt and linear
        :param object_list: target objects, will be checked by rrt, but not by linear
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        app_mot_data = self.gen_approach_to_given_conf(goal_jnt_values=goal_jnt_values,
                                                       start_jnt_values=start_jnt_values,
                                                       linear_direction=approach_direction,
                                                       linear_distance=approach_distance,
                                                       ee_values=approach_ee_values,
                                                       granularity=granularity,
                                                       obstacle_list=obstacle_list,
                                                       object_list=object_list,
                                                       use_rrt=use_rrt)
        if app_mot_data is None:
            print("ADPlanner: Cannot plan the approach section of approach-depart motion with given conf!")
            return None
        else:
            dep_mot_data = self.gen_depart_from_given_conf(start_jnt_values=app_mot_data.jv_list[-1],
                                                           end_jnt_values=end_jnt_values,
                                                           linear_direction=depart_direction,
                                                           linear_distance=depart_distance,
                                                           ee_values=depart_ee_values,
                                                           granularity=granularity,
                                                           obstacle_list=obstacle_list,
                                                           object_list=object_list,
                                                           use_rrt=use_rrt)
            if dep_mot_data is None:
                print("ADPlanner: Cannot plan the depart section of approach-depart motion with given conf!")
                return None
            else:
                return app_mot_data + dep_mot_data


if __name__ == '__main__':
    import time
    from wrs import basis as rm, robot_sim as sari, robot_sim as ym, motion as mpi, motion as rrtc, modeling as gm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    goal_pos = np.array([.4, -.3, .2])
    goal_rotmat = np.eye(3)
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    goal_pos2 = np.array([.4, -.1, .2])
    goal_rotmat2 = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=goal_pos2, rotmat=goal_rotmat2).attach_to(base)
    jnt_values2 = robot.ik(tgt_pos=goal_pos2, tgt_rotmat=goal_rotmat2)
    # robot.rgt_arm.goto_given_conf(jnt_values=jnt_values)
    # robot.gen_meshmodel().attach_to(base)
    # base.run()

    adp = ADPlanner(robot)
    tic = time.time()
    # mot_data = adp.gen_approach(goal_pos, goal_rotmat, start_jnt_values=robot.get_jnt_values(),
    #                             linear_direction=np.array([0, 0, -1]), linear_distance=.5, use_rrt=False)
    # mot_data = adp.gen_depart(goal_pos, goal_rotmat, end_jnt_values=robot.get_jnt_values(),
    #                           linear_direction=np.array([0, 0, -1]), linear_distance=.1, ee_values=.05)
    # mot_data = adp.gen_approach_depart(goal_tcp_pos=goal_pos, goal_tcp_rotmat=goal_rotmat,
    #                                       start_jnt_values=robot.get_jnt_values(),
    #                                       end_jnt_values=robot.get_jnt_values(),
    #                                       approach_ee_values=.05, depart_ee_values=.01)
    mot_data = adp.gen_depart_approach_with_given_conf(start_jnt_values=jnt_values, end_jnt_values=jnt_values2,
                                                       depart_ee_values=.03, approach_ee_values=.01, use_rrt=False)


    class Data(object):
        def __init__(self, mot_data):
            self.counter = 0
            self.mot_data = mot_data


    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()

    # Plot each column as a separate curve
    for i in range(robot.n_dof):
        ax.plot(np.asarray(mot_data.jv_list)[:, i], label=f'Column {i + 1}')
    plt.show()

    anime_data = Data(mot_data)


    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            # for mesh_model in anime_data.mot_data.mesh_list:
            #     mesh_model.detach()
            anime_data.counter = 0
        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(base)
        if base.inputmgr.keymap["space"]:
            anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.03, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)
    base.run()

    # for i, jnt_values in enumerate(conf_list):
    #     robot.goto_given_conf(jnt_values)
    #     robot.change_jaw_width(jaw_width_list[i])
    #     robot.gen_meshmodel().attach_to(base)
    #     # robot.show_cdprim()
    # base.run()
