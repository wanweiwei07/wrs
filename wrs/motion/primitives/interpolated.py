import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.motion.motion_data as motd
import wrs.modeling.geometric_model as mgm


class InterplatedMotion(object):
    """
    NOTE: only accept robot as initiator
    author: weiwei
    date: 20230809
    """

    def __init__(self, robot):
        if not isinstance(robot, sari.SglArmRobotInterface):
            if not hasattr(robot, "delegator") or not isinstance(robot.delegator, sari.SglArmRobotInterface):
                raise ValueError("Only single arm robot can be used to initiate an InterplateMotion instance!")
        self.robot = robot
        # define data type
        self.toggle_keep = True
        self.toggle_off_eecd = True

    @staticmethod
    def keep_states_decorator(method):
        """
        decorator function for save and restore robot states
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, *args, **kwargs):
            if getattr(self, "toggle_keep", True):
                self.robot.backup_state()
                result = method(self, *args, **kwargs)
                self.robot.restore_state()
                return result
            else:
                result = method(self, *args, **kwargs)
                return result

        return wrapper

    @staticmethod
    def keep_eecd_decorator(method):
        """
        decorator function for save and restore robot states
        :return:
        author: weiwei
        date: 20220404
        """

        def wrapper(self, *args, **kwargs):
            if getattr(self, "toggle_off_eecd", True):
                self.robot.toggle_off_eecd()
                result = method(self, *args, **kwargs)
                self.robot.toggle_on_eecd()
                return result
            else:
                result = method(self, *args, **kwargs)
                return result

        return wrapper

    @keep_states_decorator
    @keep_eecd_decorator
    def gen_linear_motion(self,
                          start_tcp_pos,
                          start_tcp_rotmat,
                          goal_tcp_pos,
                          goal_tcp_rotmat,
                          obstacle_list=None,
                          granularity=0.03,
                          ee_values=None,
                          toggle_dbg=False):
        """
        :param start_tcp_pos:
        :param start_tcp_rotmat:
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param ee_values: end_effector values
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
        jv_list = []
        ev_list = []
        mesh_list = []
        seed_jnt_values = None
        for i, pose in enumerate(pose_list):
            pos, rotmat = pose
            jnt_values = self.robot.ik(pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                if toggle_dbg:
                    for jnt_values in jv_list:
                        self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                        self.robot.gen_meshmodel(alpha=.3).attach_to(base)
                    base.run()
                print("IK not solvable in gen_linear_motion!")
                return None
            else:
                self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
                result, contacts = self.robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=True)
                if result:
                    if toggle_dbg:
                        for pnt in contacts:
                            mgm.gen_sphere(pnt, radius=.005).attach_to(base)
                        print(jnt_values)
                        self.robot.goto_given_conf(jnt_values=jnt_values)
                        if ee_values is not None:
                            self.robot.change_jaw_width(jaw_width=ee_values)
                        self.robot.gen_meshmodel(alpha=.3).attach_to(base)
                        base.run()
                    print("Intermediated pose collided in gen_linear_motion!")
                    return None
            jv_list.append(jnt_values)
            ev_list.append(self.robot.get_ee_values())
            if getattr(base, "toggle_mesh", True):
                mesh_list.append(self.robot.gen_meshmodel())
            seed_jnt_values = jnt_values
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=jv_list, ev_list=ev_list, mesh_list=mesh_list)
        return mot_data

    @keep_states_decorator
    def gen_interplated_between_given_conf(self,
                                           start_jnt_values,
                                           end_jnt_values,
                                           obstacle_list=None,
                                           granularity=0.03,
                                           keep_state=True,  # for decorator
                                           ee_values=None):
        interpolated_jnt_values = rm.interpolate_vectors(start_vector=start_jnt_values,
                                                         end_vector=end_jnt_values,
                                                         granularity=granularity)
        clipped_interplation = np.clip(interpolated_jnt_values, self.robot.jnt_ranges[:, 0],
                                       self.robot.jnt_ranges[:, 1])
        jv_list = []
        ev_list = []
        mesh_list = []
        for jnt_values in clipped_interplation:
            self.robot.goto_given_conf(jnt_values=jnt_values, ee_values=ee_values)
            result, contacts = self.robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=True)
            if result:
                print("Intermediated conf collided in gen_interpolated_motion!")
                return None
            jv_list.append(jnt_values)
            ev_list.append(self.robot.get_ee_values())
            if getattr(base, "toggle_mesh", True):
                mesh_list.append(self.robot.gen_meshmodel())
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=jv_list, ev_list=ev_list, mesh_list=mesh_list)
        return mot_data

    def gen_rel_linear_motion(self,
                              goal_tcp_pos,
                              goal_tcp_rotmat,
                              direction=None,
                              distance=.07,
                              obstacle_list=None,
                              granularity=0.03,
                              type="sink",
                              ee_values=None,
                              toggle_dbg=False):
        """
        generate relative linear motion considering a given goal
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param direction:
        :param distance:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param type: "sink"", or "source", motion will be ending at goal if "sink", and starting at goal if "source"
        :param ee_values: end_effector values
        :param toggle_dbg:
        :return: conf_list
        author: weiwei
        date: 20210114
        """
        if type == "sink":
            if direction is None:
                direction = goal_tcp_rotmat[:, 2]
            start_tcp_pos = goal_tcp_pos - rm.unit_vector(direction) * distance
            start_tcp_rotmat = goal_tcp_rotmat
            return self.gen_linear_motion(start_tcp_pos=start_tcp_pos,
                                          start_tcp_rotmat=start_tcp_rotmat,
                                          goal_tcp_pos=goal_tcp_pos,
                                          goal_tcp_rotmat=goal_tcp_rotmat,
                                          obstacle_list=obstacle_list,
                                          granularity=granularity,
                                          ee_values=ee_values,
                                          toggle_dbg=toggle_dbg)
        elif type == "source":
            if direction is None:
                direction = -goal_tcp_rotmat[:, 2]
            start_tcp_pos = goal_tcp_pos
            start_tcp_rotmat = goal_tcp_rotmat
            goal_tcp_pos = start_tcp_pos + rm.unit_vector(direction) * distance
            goal_tcp_rotmat = start_tcp_rotmat
            return self.gen_linear_motion(start_tcp_pos=start_tcp_pos,
                                          start_tcp_rotmat=start_tcp_rotmat,
                                          goal_tcp_pos=goal_tcp_pos,
                                          goal_tcp_rotmat=goal_tcp_rotmat,
                                          obstacle_list=obstacle_list,
                                          granularity=granularity,
                                          ee_values=ee_values,
                                          toggle_dbg=toggle_dbg)
        else:
            raise ValueError("Type must be sink or source!")

    @keep_states_decorator
    @keep_eecd_decorator
    def gen_rel_linear_motion_with_given_conf(self,
                                              goal_jnt_values,
                                              direction=None,
                                              distance=.07,
                                              obstacle_list=None,
                                              granularity=0.02,
                                              type="sink",
                                              ee_values=None,
                                              toggle_dbg=True):
        """
        :param goal_jnt_values:
        :param direction:
        :param distance:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param type: "sink", or "source", motion will be ending at goal if "sink", and starting at goal if "source"
        :param ee_values: end_effector values
        :param toggle_dbg:
        :return:
        author: weiwei
        date: 20210114
        """
        goal_tcp_pos, goal_tcp_rotmat = self.robot.fk(jnt_values=goal_jnt_values)
        if type == "sink":
            if direction is None:
                direction = goal_tcp_rotmat[:, 2]
            start_tcp_pos = goal_tcp_pos - rm.unit_vector(direction) * distance
            start_tcp_rotmat = goal_tcp_rotmat
        elif type == "source":
            if direction is None:
                direction = -goal_tcp_rotmat[:, 2]
            start_tcp_pos = goal_tcp_pos
            start_tcp_rotmat = goal_tcp_rotmat
            goal_tcp_pos = start_tcp_pos + rm.unit_vector(direction) * distance
            goal_tcp_rotmat = goal_tcp_rotmat
        else:
            raise ValueError("Type must be sink or source!")
        pose_list = rm.interplate_pos_rotmat(start_pos=start_tcp_pos,
                                             start_rotmat=start_tcp_rotmat,
                                             goal_pos=goal_tcp_pos,
                                             goal_rotmat=goal_tcp_rotmat,
                                             granularity=granularity)
        jv_list = []
        ev_list = []
        mesh_list = []
        seed_jnt_values = goal_jnt_values
        for pos, rotmat in pose_list:
            jnt_values = self.robot.ik(pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                if toggle_dbg:
                    for jnt_values in jv_list:
                        self.robot.goto_given_conf(jnt_values, ee_values=ee_values)
                        self.robot.gen_meshmodel(alpha=.3).attach_to(base)
                    base.run()
                print("IK not solvable in gen_linear_motion!")
                return None
            else:
                self.robot.goto_given_conf(jnt_values, ee_values=ee_values)
                result, contacts = self.robot.is_collided(obstacle_list=obstacle_list, toggle_contacts=True)
                if result:
                    if toggle_dbg:
                        for pnt in contacts:
                            mgm.gen_sphere(pnt, radius=.005).attach_to(base)
                        print(jnt_values)
                        self.robot.goto_given_conf(jnt_values)
                        if ee_values is not None:
                            self.robot.change_jaw_width(jaw_width=ee_values)
                        self.robot.gen_meshmodel(toggle_cdprim=True, alpha=.3).attach_to(base)
                        base.run()
                    print("Intermediated pose collided in gen_linear_motion!")
                    return None
            jv_list.append(jnt_values)
            ev_list.append(self.robot.get_ee_values())
            if getattr(base, "toggle_mesh", True):
                mesh_list.append(self.robot.gen_meshmodel())
            seed_jnt_values = jnt_values
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=jv_list, ev_list=ev_list, mesh_list=mesh_list)
        return mot_data

    @keep_states_decorator
    def gen_circular_motion(self,
                            circle_center_pos,
                            circle_normal_ax,
                            start_tcp_rotmat,
                            end_tcp_rotmat,
                            radius=.02,
                            obstacle_list=None,
                            granularity=0.03,
                            ee_values=None,
                            toggle_dbg=False):
        """
        :param circle_center_pos
        :param circle_normal_ax
        :param start_tcp_pos:
        :param start_tcp_rotmat:
        :param goal_tcp_pos:
        :param goal_tcp_rotmat:
        :param goal_info:
        :param obstacle_list:
        :param granularity: resolution of steps in workspace
        :param ee_values: end_effector values
        :return:
        author: weiwei
        date: 20210501
        """
        pose_list = rm.interplate_pos_rotmat_around_circle(circle_center_pos, circle_normal_ax, radius,
                                                           start_tcp_rotmat, end_tcp_rotmat, granularity)
        jv_list = []
        ev_list = []
        mesh_list = []
        seed_jnt_values = None
        for pos, rotmat in pose_list:
            jnt_values = self.robot.ik(pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("IK not solvable in gen_circular_motion!")
                return None
            else:
                self.robot.goto_given_conf(jnt_values, ee_values=ee_values)
                result, contacts = self.robot.is_collided(obstacle_list, toggle_contacts=True)
                if result:
                    if toggle_dbg:
                        for pnt in contacts:
                            mgm.gen_sphere(pnt, radius=.005).attach_to(base)
                    print("Intermediate pose collided in gen_linear_motion!")
                    return None
            jv_list.append(jnt_values)
            ev_list.append(self.robot.get_ee_values())
            if getattr(base, "toggle_mesh", True):
                mesh_list.append(self.robot.gen_meshmodel())
            seed_jnt_values = jnt_values
        mot_data = motd.MotionData(robot=self.robot)
        mot_data.extend(jv_list=jv_list, ev_list=ee_values, mesh_list=mesh_list)
        return mot_data


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.robots.yumi.yumi as ym

    base = wd.World(cam_pos=[3, 2, 2], lookat_pos=[0, 0, 0.2])
    mgm.gen_frame().attach_to(base)
    robot = ym.Yumi(enable_cc=True)
    robot.use_rgt()
    start_pos = np.array([.55, -.1, .4])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_pos = np.array([.55, -.1, .3])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    mgm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    interplator = InterplatedMotion(robot)
    # tic = time.time()
    # conf_list = interplator.gen_linear_motion(start_tcp_pos=start_pos, start_tcp_rotmat=start_rotmat,
    #                                           goal_tcp_pos=goal_pos, goal_tcp_rotmat=goal_rotmat,
    #                                           toggle_dbg=False)
    # toc = time.time()
    # print(toc - tic)
    # for jnt_values in conf_list:
    #     robot.goto_given_conf(jnt_values)
    #     robot.gen_meshmodel(alpha=.3).attach_to(base)
    # base.run()

    tic = time.time()
    start_conf = robot.ik(tgt_pos=start_pos, tgt_rotmat=start_rotmat)
    mot_data = interplator.gen_rel_linear_motion_with_given_conf(start_conf,
                                                                 direction=goal_pos - start_pos,
                                                                 distance=.1,
                                                                 obstacle_list=None,
                                                                 granularity=0.03,
                                                                 type="source",
                                                                 toggle_dbg=False)
    print(mot_data)
    toc = time.time()
    print(toc - tic)
    for i, jnt_values in enumerate(mot_data):
        mesh_model = mot_data.mesh_list[i]
        mesh_model.rgb = rm.bc.cool_map(i / len(mot_data))
        mesh_model.alpha = .3
        mesh_model.attach_to(base)
    base.run()
