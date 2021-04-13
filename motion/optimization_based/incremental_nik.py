import math
import numpy as np
import basis.robot_math as rm


class IncrementalNIK(object):

    def __init__(self, robot_sim):
        self.rbt = robot_sim

    def gen_linear_motion(self,
                          component_name,
                          start_hnd_pos,
                          start_hnd_rotmat,
                          goal_hnd_pos,
                          goal_hnd_rotmat,
                          obstacle_list=[],
                          granularity=0.03,
                          seed_jnt_values=None):
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
        jnt_values_bk = self.rbt.get_jnt_values(component_name)
        pos_list, rotmat_list = rm.interplate_pos_rotmat(start_hnd_pos,
                                                         start_hnd_rotmat,
                                                         goal_hnd_pos,
                                                         goal_hnd_rotmat,
                                                         granularity=granularity)
        jnt_values_list = []
        if seed_jnt_values is None:
            seed_jnt_values = jnt_values_bk
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.rbt.ik(component_name, pos, rotmat, seed_jnt_values=seed_jnt_values)
            if jnt_values is None:
                print("IK not solvable in gen_linear_motion!")
                self.rbt.fk(component_name, jnt_values_bk)
                return []
            else:
                self.rbt.fk(component_name, jnt_values)
                if self.rbt.is_collided(obstacle_list):
                    print("Intermediate pose collided in gen_linear_motion!")
                    self.rbt.fk(component_name, jnt_values_bk)
                    return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.rbt.fk(component_name, jnt_values_bk)
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
                              type='sink'):
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
                                          seed_jnt_values)
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
                                          seed_jnt_values)
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
