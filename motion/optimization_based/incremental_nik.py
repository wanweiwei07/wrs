import math
import numpy as np
import basis.robot_math as rm
import modeling.geometricmodel as gm


class IncrementalNIK(object):

    def __init__(self, robot_s):
        self.robot_s = robot_s

    def gen_linear_motion(self,
                          component_name,
                          start_hnd_pos,
                          start_hnd_rotmat,
                          goal_hnd_pos,
                          goal_hnd_rotmat,
                          obstacle_list=[],
                          granularity=0.03,
                          seed_jnt_values=None,
                          toggle_debug=False):
        """
        :param component_name:
        :param start_hnd_pos:
        :param start_hnd_rotmat:
        :param goal_hnd_pos:
        :param goal_hnd_rotmat:
        :param goal_info:
        :param obstacle_list:
        :param granularity:
        :return:
        author: weiwei
        date: 20210125
        """
        jnt_values_bk = self.robot_s.get_jnt_values(component_name)
        pos_list, rotmat_list = rm.interplate_pos_rotmat(start_hnd_pos,
                                                         start_hnd_rotmat,
                                                         goal_hnd_pos,
                                                         goal_hnd_rotmat,
                                                         granularity=granularity)
        jnt_values_list = []
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.robot_s.ik(component_name, pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("IK not solvable in gen_linear_motion!")
                self.robot_s.fk(component_name, jnt_values_bk)
                return []
            else:
                self.robot_s.fk(component_name, jnt_values)
                cd_result, ct_points = self.robot_s.is_collided(obstacle_list, toggle_contact_points=True)
                if cd_result:
                    if toggle_debug:
                        for ct_pnt in ct_points:
                            gm.gen_sphere(ct_pnt).attach_to(base)
                    print("Intermediate pose collided in gen_linear_motion!")
                    self.robot_s.fk(component_name, jnt_values_bk)
                    return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.robot_s.fk(component_name, jnt_values_bk)
        return jnt_values_list

    def gen_rel_linear_motion(self,
                              component_name,
                              goal_hnd_pos,
                              goal_hnd_rotmat,
                              direction,
                              distance,
                              obstacle_list=[],
                              granularity=0.03,
                              seed_jnt_values=None,
                              type='sink',
                              toggle_debug=False):
        """
        :param goal_info:
        :param direction:
        :param distance:
        :param obstacle_list:
        :param granularity:
        :param type: 'sink', or 'source'
        :return:
        author: weiwei
        date: 20210114
        """
        if type == 'sink':
            start_hnd_pos = goal_hnd_pos - rm.unit_vector(direction) * distance
            start_hnd_rotmat = goal_hnd_rotmat
            return self.gen_linear_motion(component_name,
                                          start_hnd_pos,
                                          start_hnd_rotmat,
                                          goal_hnd_pos,
                                          goal_hnd_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        elif type == 'source':
            start_hnd_pos = goal_hnd_pos
            start_hnd_rotmat = goal_hnd_rotmat
            goal_hnd_pos = goal_hnd_pos + direction * distance
            goal_hnd_rotmat = goal_hnd_rotmat
            return self.gen_linear_motion(component_name,
                                          start_hnd_pos,
                                          start_hnd_rotmat,
                                          goal_hnd_pos,
                                          goal_hnd_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        else:
            raise ValueError("Type must be sink or source!")

    def gen_rel_linear_motion_with_given_conf(self,
                                              component_name,
                                              goal_jnt_values,
                                              direction,
                                              distance,
                                              obstacle_list=[],
                                              granularity=0.03,
                                              seed_jnt_values=None,
                                              type='sink',
                                              toggle_debug=False):
        """
        :param goal_info:
        :param direction:
        :param distance:
        :param obstacle_list:
        :param granularity:
        :param type: 'sink', or 'source'
        :return:
        author: weiwei
        date: 20210114
        """
        goal_hnd_pos, goal_hnd_rotmat = self.robot_s.cvt_conf_to_tcp(component_name, goal_jnt_values)
        if type == 'sink':
            start_hnd_pos = goal_hnd_pos - rm.unit_vector(direction) * distance
            start_hnd_rotmat = goal_hnd_rotmat
            return self.gen_linear_motion(component_name,
                                          start_hnd_pos,
                                          start_hnd_rotmat,
                                          goal_hnd_pos,
                                          goal_hnd_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        elif type == 'source':
            start_hnd_pos = goal_hnd_pos
            start_hnd_rotmat = goal_hnd_rotmat
            goal_hnd_pos = goal_hnd_pos + direction * distance
            goal_hnd_rotmat = goal_hnd_rotmat
            return self.gen_linear_motion(component_name,
                                          start_hnd_pos,
                                          start_hnd_rotmat,
                                          goal_hnd_pos,
                                          goal_hnd_rotmat,
                                          obstacle_list,
                                          granularity,
                                          seed_jnt_values,
                                          toggle_debug=toggle_debug)
        else:
            raise ValueError("Type must be sink or source!")

    def get_rotational_motion(self,
                              component_name,
                              start_hnd_pos,
                              start_hnd_rotmat,
                              goal_hnd_pos,
                              goal_hnd_rotmat,
                              obstacle_list=[],
                              rot_center=np.zeros(3),
                              rot_axis=np.array([1, 0, 0]),
                              granularity=0.03,
                              seed_jnt_values=None):
        # TODO
        pass


if __name__ == '__main__':
    import time
    import robotsim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(cam_pos=[1.5, 0, 3], lookat_pos=[0, 0, .5])
    gm.gen_frame().attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    component_name = 'rgt_arm'
    start_pos = np.array([.5, -.3, .3])
    start_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    goal_pos = np.array([.55, .3, .5])
    goal_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi / 2)
    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    inik = IncrementalNIK(yumi_instance)
    tic = time.time()
    jnt_values_list = inik.gen_linear_motion(component_name, start_hnd_pos=start_pos, start_hnd_rotmat=start_rotmat,
                                             goal_hnd_pos=goal_pos, goal_hnd_rotmat=goal_rotmat)
    toc = time.time()
    print(toc - tic)
    for jnt_values in jnt_values_list:
        yumi_instance.fk(component_name, jnt_values)
        yumi_meshmodel = yumi_instance.gen_meshmodel()
        yumi_meshmodel.attach_to(base)
    base.run()
