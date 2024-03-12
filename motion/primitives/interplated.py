import math
import numpy as np
import basis.robot_math as rm
import motion.utils as utils
import robot_sim.robots.single_arm_robot_interface as sari


class InterplatedMotion(object):
    """
    NOTE: only accept sgl_arm_robot as initiator
    author: weiwei
    date: 20230809
    """

    def __init__(self, sgl_arm_robot):
        if not isinstance(sgl_arm_robot, sari.SglArmRobotInterface):
            raise ValueError("Only single arm robot can be used to initiate an InterplateMotion instance!")
        self.robot = sgl_arm_robot

    @utils.keep_jnt_values_decorator
    def gen_linear_motion(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          goal_tcp_pos,
                          goal_tcp_rotmat,
                          obstacle_list=[],
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_dbg=False):
        """
        :param start_tcp_pos:
        :param start_tcp_rotmat:
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param seed_jnt_values
        :param toggle_dbg
        :return:
        author: weiwei
        date: 20210125
        """
        pose_list = rm.interplate_pos_rotmat(start_tcp_pos,
                                             start_tcp_rotmat,
                                             goal_tcp_pos,
                                             goal_tcp_rotmat,
                                             granularity=granularity)
        jnt_values_list = []
        for pos, rotmat in pose_list:
            jnt_values = self.robot.ik(pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                if toggle_dbg:
                    for jnt_values in jnt_values_list:
                        self.robot.goto_given_conf(jnt_values)
                        self.robot.gen_meshmodel(alpha=.3).attach_to(base)
                    base.run()
                print("IK not solvable in gen_linear_motion!")
                return None
            else:
                self.robot.goto_given_conf(jnt_values)
                result, contacts = self.robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=True)
                if result:
                    if toggle_dbg:
                        for pnt in contacts:
                            gm.gen_sphere(pnt, radius=.005).attach_to(base)
                        print(jnt_values)
                        self.robot.goto_given_conf(jnt_values)
                        self.robot.gen_meshmodel(alpha=.3).attach_to(base)
                        base.run()
                    print("Intermediated pose collided in gen_linear_motion!")
                    return None
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        return jnt_values_list

    def gen_rel_linear_motion(self,
                              goal_tcp_pos,
                              goal_tcp_rotmat,
                              motion_vec,
                              motion_dist,
                              obstacle_list=[],
                              granularity=0.03,
                              seed_jnt_values=None,
                              type='sink',
                              toggle_dbg=False):
        """
        generate relative linear motion considering a given goal
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param motion_vec:
        :param motion_dist:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param seed_jnt_values:
        :param type: 'sink', or 'source', motion will be ending at goal if "sink", and starting at goal if "source"
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20210114
        """
        if type == 'sink':
            start_tcp_pos = goal_tcp_pos - rm.unit_vector(motion_vec) * motion_dist
            start_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_dbg)
        elif type == 'source':
            start_tcp_pos = goal_tcp_pos
            start_tcp_rotmat = goal_tcp_rotmat
            goal_tcp_pos = goal_tcp_pos + motion_vec * motion_dist
            goal_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(start_tcp_pos,
                                          start_tcp_rotmat,
                                          goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_dbg)
        else:
            raise ValueError("Type must be sink or source!")

    def gen_rel_linear_motion_with_given_conf(self,
                                              goal_jnt_values,
                                              motion_vec,
                                              motion_dist,
                                              obstacle_list=[],
                                              granularity=0.03,
                                              seed_jnt_values=None,
                                              type='sink',
                                              toggle_dbg=False):
        """
        :param goal_jnt_values:
        :param motion_vec:
        :param motion_dist:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param seed_jnt_values:
        :param type: 'sink', or 'source', motion will be ending at goal if "sink", and starting at goal if "source"
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20210114
        """
        goal_tcp_pos, goal_tcp_rotmat = self.robot.fk(goal_jnt_values)
        return self.gen_rel_linear_motion(goal_tcp_pos,
                                          goal_tcp_rotmat,
                                          motion_vec,
                                          motion_dist,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          type,
                                          toggle_dbg)

    def get_rotational_motion(self,
                              start_tcp_pos,
                              start_tcp_rotmat,
                              goal_tcp_pos,
                              goal_tcp_rotmat,
                              obstacle_list,
                              rot_center=np.zeros(3),
                              rot_axis=np.array([1, 0, 0]),
                              granularity=0.03,
                              seed_jnt_values=None):
        raise NotImplementedError

    @utils.keep_jnt_values_decorator
    def gen_circular_motion(self,
                            circle_center_pos,
                            circle_ax,
                            start_tcp_rotmat,
                            end_tcp_rotmat,
                            radius=.02,
                            obstacle_list=[],
                            granularity=0.03,
                            seed_jnt_values=None,
                            toggle_tcp_list=False,
                            toggle_dbg=False):
        """
        :param component_name:
        :param start_tcp_pos:
        :param start_tcp_rotmat:
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param goal_info:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :return:
        author: weiwei
        date: 20210501
        """
        pose_list = rm.interplate_pos_rotmat_around_circle(circle_center_pos,
                                                           circle_ax,
                                                           radius,
                                                           start_tcp_rotmat,
                                                           end_tcp_rotmat,
                                                           granularity)
        jnt_values_list = []
        for pos, rotmat in pose_list:
            jnt_values = self.robot.ik(pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("IK not solvable in gen_circular_motion!")
                return None
            else:
                self.robot.goto_given_conf(jnt_values)
                result, contacts = self.robot.is_collided(obstacle_list, toggle_contacts=True)
                if result:
                    if toggle_dbg:
                        for pnt in contacts:
                            gm.gen_sphere(pnt, radius=.005).attach_to(base)
                    print("Intermediate pose collided in gen_linear_motion!")
                    return None
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        if toggle_tcp_list:
            return jnt_values_list, pose_list
        else:
            return jnt_values_list


if __name__ == '__main__':
    import time
    import robot_sim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[3, 2, 2], lookat_pos=[0, 0, 0.2])
    gm.gen_frame().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    # start_pos = np.array([.5, -.3, .3])
    # start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    # goal_pos = np.array([.55, -.2, .6])
    # goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    start_pos = np.array([.55, -.1, .4])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_pos = np.array([.55, -.1, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    interplator = InterplatedMotion(robot.rgt_arm)
    tic = time.time()
    jnt_values_list = interplator.gen_linear_motion(start_tcp_pos=start_pos, start_tcp_rotmat=start_rotmat,
                                                    goal_tcp_pos=goal_pos, goal_tcp_rotmat=goal_rotmat,
                                                    toggle_dbg=False)
    toc = time.time()
    print(toc - tic)
    for jnt_values in jnt_values_list:
        robot.rgt_arm.goto_given_conf(jnt_values)
        robot.gen_meshmodel(alpha=.3).attach_to(base)
    base.run()
