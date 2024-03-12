import os
import math
import copy
import pickle
import numpy as np
import basis.data_adapter as da
import modeling.collision_model as cm
import motion.primitives.interplated as mpi
import motion.probabilistic.rrt_connect as rrtc
import robot_sim.robots.single_arm_robot_interface as sari


class ADPlanner(object):
    """
    AD = Approach_Depart
    NOTE: only accept sgl_arm_robot as initiator
    """

    def __init__(self, sgl_arm_robot):
        """
        :param robot_s:
        author: weiwei, hao
        date: 20191122, 20210113
        """
        if not isinstance(sgl_arm_robot, sari.SglArmRobotInterface):
            raise ValueError("Only single arm robot can be used to initiate an InterplateMotion instance!")
        self.robot = sgl_arm_robot
        self.rrtc_planner = rrtc.RRTConnect(self.robot)
        self.im_planner = mpi.InterplatedMotion(self.robot)

    def gen_jaw_width_list(self, conf_list, jaw_width):
        jaw_width_list = [jaw_width]*len(conf_list)
        return jaw_width_list

    def gen_approach_linear(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            approaching_vec=None,
                            approaching_dist=.1,
                            approaching_jaw_width=.05,
                            granularity=0.03,
                            obstacle_list=[],
                            seed_jnt_values=None,
                            toggle_end_jaw_motion=False,
                            end_jaw_width=.0):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param approaching_vec: use the loc_z of goal_tcp_rotmat if None
        :param approaching_dist:
        :param approaching_jaw_width:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values
        :param toggle_end_jaw_motion:
        :param end_jaw_width: only used when toggle_end_jaw_motion is True
        :return:
        author: weiwei
        date: 20210125
        """
        if approaching_vec is None:
            approaching_vec = goal_tcp_rotmat[:, 2]
        conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                          goal_tcp_rotmat,
                                                          approaching_vec,
                                                          approaching_dist,
                                                          obstacle_list=obstacle_list,
                                                          granularity=granularity,
                                                          type='sink',
                                                          seed_jnt_values=seed_jnt_values)
        if conf_list is None:
            print('Cannot perform approach linear!')
            return None, None
        else:
            if toggle_end_jaw_motion:
                jawwidth_list = self.gen_jaw_width_list(conf_list, approaching_jaw_width)
                conf_list += [conf_list[-1]]
                jawwidth_list += [end_jaw_width]
                return conf_list, jawwidth_list
            else:
                return conf_list, self.gen_jaw_width_list(conf_list, approaching_jaw_width)

    def gen_depart_linear(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          depart_direction=None,  # np.array([0, 0, 1])
                          depart_distance=.1,
                          depart_jaw_width=.05,
                          granularity=0.03,
                          obstacle_list=[],
                          seed_jnt_values=None,
                          toggle_start_jaw_motion=False,
                          start_jaw_width=.0):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param seed_jnt_values:
        :param toggle_start_jaw_motion:
        :param start_jaw_width: only used when toggle_end_jaw_motion is True
        :return: conf_list, jaw_width_list, objhomomat_list_list
        author: weiwei
        date: 20210125
        """
        if depart_direction is None:
            depart_direction = -start_tcp_rotmat[:, 2]
        conf_list = self.im_planner.gen_rel_linear_motion(start_tcp_pos,
                                                          start_tcp_rotmat,
                                                          depart_direction,
                                                          depart_distance,
                                                          obstacle_list=obstacle_list,
                                                          granularity=granularity,
                                                          type='source',
                                                          seed_jnt_values=seed_jnt_values)
        if conf_list is None:
            print('Cannot perform depart action!')
            return None, None
        else:
            if toggle_start_jaw_motion:
                jaw_width_list = self.gen_jaw_width_list(conf_list, depart_jaw_width)
                conf_list = [conf_list[0]] + conf_list
                jaw_width_list = [start_jaw_width] + jaw_width_list
                return conf_list, jaw_width_list
            else:
                return conf_list, self.gen_jaw_width_list(conf_list, depart_jaw_width)

    def gen_approach_and_depart_linear(self,
                                       goal_tcp_pos,
                                       goal_tcp_rotmat,
                                       approach_direction=None,  # np.array([0, 0, -1])
                                       approach_distance=.1,
                                       approach_jaw_width=.05,
                                       depart_direction=None,  # np.array([0, 0, 1])
                                       depart_distance=.1,
                                       depart_jawwidth=0,
                                       granularity=.03,
                                       obstacle_list=[],
                                       seed_jnt_values=None):
        """
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param hnd_name:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jawwidth:
        :param granularity:
        :param obstacle_list:
        :param seed_jnt_values
        :return: approach_conf_list, depart_jaw_width_list
        author: weiwei, hao
        date: 20191122, 20200105, 20210113, 20210125
        """
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]
        approach_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                                   goal_tcp_rotmat,
                                                                   approach_direction,
                                                                   approach_distance,
                                                                   obstacle_list=obstacle_list,
                                                                   granularity=granularity,
                                                                   type='sink',
                                                                   seed_jnt_values=seed_jnt_values)
        if approach_conf_list is None:
            print("ADPlanner: Cannot gen approach linear!")
            return None, None
        if approach_distance != 0 and len(approach_conf_list) == 0:
            print('Cannot perform approach action!')
        else:
            if depart_direction is None:
                depart_direction = goal_tcp_rotmat[:, 2]
            depart_conf_list = self.im_planner.gen_rel_linear_motion(goal_tcp_pos,
                                                                     goal_tcp_rotmat,
                                                                     depart_direction,
                                                                     depart_distance,
                                                                     obstacle_list=obstacle_list,
                                                                     granularity=granularity,
                                                                     type='source',
                                                                     seed_jnt_values=approach_conf_list[-1])
            if depart_conf_list is None:
                print("ADPlanner: Cannot gen depart linear!")
                return None, None
            if depart_distance != 0 and len(depart_conf_list) == 0:
                print('Cannot perform depart action!')
            else:
                approach_jaw_width_list = self.gen_jaw_width_list(approach_conf_list, approach_jaw_width)
                depart_jaw_width_list = self.gen_jaw_width_list(depart_conf_list, depart_jawwidth)
                return approach_conf_list + depart_conf_list, approach_jaw_width_list + depart_jaw_width_list
        return [], []

    def gen_approach_motion(self,
                            goal_tcp_pos,
                            goal_tcp_rotmat,
                            start_conf=None,
                            approach_direction=None,
                            approach_distance=.1,
                            approach_jaw_width=.05,
                            granularity=.03,
                            obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                            object_list=[],  # target objects, will be checked by rrt, but not by linear
                            seed_jnt_values=None,
                            toggle_end_jaw_motion=False,
                            end_jaw_width=.0):
        if seed_jnt_values is None:
            seed_jnt_values = start_conf
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]
        conf_list, jaw_width_list = self.gen_approach_linear(goal_tcp_pos,
                                                             goal_tcp_rotmat,
                                                             approach_direction,
                                                             approach_distance,
                                                             approach_jaw_width,
                                                             granularity,
                                                             obstacle_list,
                                                             seed_jnt_values,
                                                             toggle_end_jaw_motion,
                                                             end_jaw_width)
        if conf_list is None:
            print("ADPlanner: Cannot gen approach linear!")
            return None, None
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_planner.plan(start_conf=start_conf,
                                                              goal_conf=conf_list[0],
                                                              obstacle_list=obstacle_list + object_list,
                                                              ext_dist=.05,
                                                              max_time=100)
            if start2approach_conf_list is None:
                print("ADPlanner: Cannot plan the motion from start_conf to the beginning of approach!")
                return None, None
            start2approach_jaw_width_list = self.gen_jaw_width_list(start2approach_conf_list, approach_jaw_width)
        return start2approach_conf_list + conf_list, start2approach_jaw_width_list + jaw_width_list

    def gen_depart_motion(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          end_conf=None,
                          depart_direction=None,
                          depart_distance=.1,
                          depart_jaw_width=.05,
                          granularity=.03,
                          obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                          object_list=[],  # target objects, will be checked by rrt, but not by linear
                          seed_jnt_values=None,
                          toggle_start_jaw_motion=False,
                          begin_jaw_width=.0):
        if seed_jnt_values is None:
            seed_jnt_values = end_conf
        if depart_direction is None:
            depart_direction = start_tcp_rotmat[:, 2]
        conf_list, jaw_width_list = self.gen_depart_linear(start_tcp_pos,
                                                           start_tcp_rotmat,
                                                           depart_direction,
                                                           depart_distance,
                                                           depart_jaw_width,
                                                           granularity,
                                                           obstacle_list,
                                                           seed_jnt_values,
                                                           toggle_start_jaw_motion,
                                                           begin_jaw_width)
        if conf_list is None:
            print("ADPlanner: Cannot gen depart linear!")
            return None, None
        if end_conf is not None:
            depart2goal_conf_list = self.rrtc_planner.plan(start_conf=conf_list[-1],
                                                           goal_conf=end_conf,
                                                           obstacle_list=obstacle_list + object_list,
                                                           ext_dist=.05,
                                                           max_time=100)
            if depart2goal_conf_list is None:
                print("ADPlanner: Cannot plan depart motion!")
                return None, None
            depart2goal_jaw_width_list = self.gen_jaw_width_list(depart2goal_conf_list, depart_jaw_width)
        else:
            depart2goal_conf_list = []
            depart2goal_jaw_width_list = []
        return conf_list + depart2goal_conf_list, jaw_width_list + depart2goal_jaw_width_list

    def gen_approach_and_depart_motion(self,
                                       goal_tcp_pos,
                                       goal_tcp_rotmat,
                                       start_conf=None,
                                       goal_conf=None,
                                       approach_direction=None,  # np.array([0, 0, -1])
                                       approach_distance=.2,
                                       approach_jaw_width=.05,
                                       depart_direction=None,  # np.array([0, 0, 1])
                                       depart_distance=.2,
                                       depart_jaw_width=0,
                                       granularity=.03,
                                       obstacle_list=[],  # obstacles, will be checked by both rrt and linear
                                       object_list=[]):  # target objects, will be checked by rrt, but not by linear)
        """
        degenerate into gen_ad_primitive if both seed_jnt_values and end_conf are None
        :param component_name:
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param start_conf:
        :param goal_conf:
        :param approach_direction:
        :param approach_distance:
        :param approach_jaw_width:
        :param depart_direction:
        :param depart_distance:
        :param depart_jaw_width:
        :param granularity:
        :param seed_jnt_values:
        :param obstacle_list:
        :return:
        author: weiwei
        date: 20210113, 20210125
        """
        if approach_direction is None:
            approach_direction = goal_tcp_rotmat[:, 2]
        if depart_direction is None:
            approach_direction = -goal_tcp_rotmat[:, 2]
        ad_conf_list, ad_jaw_width_list = self.gen_approach_and_depart_linear(goal_tcp_pos,
                                                                              goal_tcp_rotmat,
                                                                              approach_direction,
                                                                              approach_distance,
                                                                              approach_jaw_width,
                                                                              depart_direction,
                                                                              depart_distance,
                                                                              depart_jaw_width,
                                                                              granularity,
                                                                              obstacle_list)
        if ad_conf_list is None:
            print("ADPlanner: Cannot gen ad linear!")
            return None, None
        if start_conf is not None:
            start2approach_conf_list = self.rrtc_planner.plan(start_conf=start_conf,
                                                              goal_conf=ad_conf_list[0],
                                                              obstacle_list=obstacle_list,
                                                              ext_dist=.05,
                                                              max_time=300)
            if start2approach_conf_list is None:
                print("ADPlanner: Cannot plan approach motion!")
                return None, None
            start2approach_jaw_width_list = self.gen_jaw_width_list(start2approach_conf_list, approach_jaw_width)
        if goal_conf is not None:
            depart2goal_conf_list = self.rrtc_planner.plan(start_conf=ad_conf_list[-1],
                                                           goal_conf=goal_conf,
                                                           obstacle_list=obstacle_list + object_list,
                                                           ext_dist=.05,
                                                           max_time=300)
            if depart2goal_conf_list is None:
                print("ADPlanner: Cannot plan depart motion!")
                return None, None
            depart2goal_jaw_width_list = self.gen_jaw_width_list(depart2goal_conf_list, depart_jaw_width)
        return (start2approach_conf_list + ad_conf_list + depart2goal_conf_list,
                start2approach_jaw_width_list + ad_jaw_width_list + depart2goal_jaw_width_list)

    def gen_depart_and_approach_linear(self):
        pass

    def gen_depart_and_approach_motion(self):
        pass


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    goal_pos = np.array([.55, -.1, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    jnt_values = robot.rgt_arm.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    # robot.rgt_arm.goto_given_conf(jnt_values=jnt_values)
    # robot.gen_meshmodel().attach_to(base)
    # base.run()

    sgl_arm = robot.rgt_arm
    adp = ADPlanner(sgl_arm)
    tic = time.time()
    conf_list, jaw_width_list = adp.gen_approach_and_depart_motion(goal_pos,
                                                                   goal_rotmat,
                                                                   start_conf=sgl_arm.get_jnt_values(),
                                                                   goal_conf=sgl_arm.get_jnt_values(),
                                                                   approach_direction=np.array([0, 0, -1]),
                                                                   approach_distance=.1,
                                                                   depart_direction=np.array([0, -1, 0]),
                                                                   depart_distance=.1,
                                                                   depart_jaw_width=0)
    # conf_list, jaw_width_list = adp.gen_approach_motion(hnd_name,
    #                                                    goal_pos,
    #                                                    goal_rotmat,
    #                                                    seed_jnt_values=robot_s.get_jnt_values(hnd_name),
    #                                                    approaching_vec=np.array([0, 0, -1]),
    #                                                    approaching_dist=.1)
    # conf_list, jaw_width_list = adp.gen_depart_motion(hnd_name,
    #                                                  goal_pos,
    #                                                  goal_rotmat,
    #                                                  end_conf=robot_s.get_jnt_values(hnd_name),
    #                                                  depart_direction=np.array([0, 0, 1]),
    #                                                  depart_distance=.1)
    toc = time.time()
    print(toc - tic)


    class Data(object):
        def __init__(self):
            self.robot_attached_list = []
            self.counter = 0
            self.conf_list =conf_list
            self.jaw_width_list = jaw_width_list
            self.robot = robot
            self.sgl_arm = robot.rgt_arm

    animation_data = Data()

    def update(animation_data, task):
        if animation_data.counter >= len(animation_data.conf_list):
            if len(animation_data.robot_attached_list) != 0:
                for robot_attached in animation_data.robot_attached_list:
                    robot_attached.detach()
            animation_data.robot_attached_list.clear()
            animation_data.counter = 0
        if len(animation_data.robot_attached_list) > 1:
            for robot_attached in animation_data.robot_attached_list:
                robot_attached.detach()
        conf = animation_data.conf_list[animation_data.counter]
        jaw_width = animation_data.jaw_width_list[animation_data.counter]
        animation_data.sgl_arm.goto_given_conf(jnt_values=conf)
        animation_data.sgl_arm.change_jaw_width(jaw_width=jaw_width)
        robot_meshmodel = animation_data.robot.gen_meshmodel(toggle_cdprim=False, alpha=1)
        robot_meshmodel.attach_to(base)
        animation_data.robot_attached_list.append(robot_meshmodel)
        animation_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[animation_data],
                          appendTask=True)
    base.run()


    # for i, jnt_values in enumerate(conf_list):
    #     sgl_arm.goto_given_conf(jnt_values)
    #     sgl_arm.change_jaw_width(jaw_width_list[i])
    #     robot.gen_meshmodel().attach_to(base)
    #     # sgl_arm.show_cdprim()
    # base.run()
