import math
import numpy as np
import basis.robotmath as rm
import warnings as wns
import robotsim._kinematics.jlchainik as jlcik

class IncrementalNIK(object):

    def __init__(self, robot, wln_ratio=.15):
        self.rbt = robot

    def _interplate(self, start_info, goal_info, granularity=.01):
        """
        :param start_info: [pos, rotmat]
        :param goal_info: [pos, rotmat]
        :param granularity
        :return: a list of 1xn nparray
        """
        start_pos, start_rotmat = start_info
        goal_pos, goal_rotmat = goal_info
        len, vec = rm.unit_vector(start_pos - goal_pos, togglelength=True)
        nval = math.ceil(len / granularity)
        pos_list = np.linspace(start_pos, goal_pos , nval)
        rotmat_list = rm.rotmat_slerp(start_rotmat, goal_rotmat, nval)
        return pos_list, rotmat_list

    def gen_linear_motion(self, jlc_name, start_info, goal_info, obstacle_list=[], granularity=0.03):
        jnt_values_bk = self.rbt.get_jnt_values(jlc_name)
        pos_list, rotmat_list = self._interplate(start_info, goal_info, granularity=granularity)
        jnt_values_list = []
        seed_jnt_values = jnt_values_bk
        for (pos, rotmat) in zip(pos_list, rotmat_list):
            jnt_values = self.rbt.ik(jlc_name, pos, rotmat, seed_conf=seed_jnt_values)
            if jnt_values is None:
                print("Not solvable!")
                self.rbt.fk(jlc_name, jnt_values_bk)
                return []
            else:
                self.rbt.fk(jlc_name, jnt_values)
                if self.rbt.is_collided(obstacle_list):
                    print("Intermediate pose collided!")
                    self.rbt.fk(jlc_name, jnt_values_bk)
                    return []
            jnt_values_list.append(jnt_values)
            seed_jnt_values = jnt_values
        self.rbt.fk(jlc_name, jnt_values_bk)
        return jnt_values_list

    def gen_rel_linear_motion(self, pose_info, direction, distance, granularity=0.03, type='sink'):
        """
        :param pose_info:
        :param direction:
        :param distance:
        :param granularity:
        :param type: 'sink', or 'source'
        :return:
        author: weiwei
        date: 20210114
        """
        if type == 'sink':
            start_info = [pose_info[0]-direction*distance, pose_info[1]]
            return self.gen_linear_motion(start_info, pose_info, granularity)
        elif type == 'source':
            goal_info = [pose_info[0]+direction*distance, pose_info[1]]
            return self.gen_linear_motion(pose_info, goal_info, granularity)
        else:
            raise  ValueError("Type must be sink or source!")

if __name__ == '__main__':
    import time
    import robotsim.robots.yumi.yumi as ym
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])
    gm.gen_frame().attach_to(base)
    yumi_instance = ym.Yumi(enable_cc=True)
    jlc_name='rgt_arm'
    start_pos = np.array([.5, -.3, .3])
    start_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2)
    start_info = [start_pos, start_rotmat]
    goal_pos = np.array([.55, .3, .5])
    goal_rotmat = rm.rotmat_from_axangle([0,1,0], math.pi/2)
    goal_info = [goal_pos, goal_rotmat]
    gm.gen_frame(pos=start_pos, rotmat=start_rotmat).attach_to(base)
    gm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    inik = IncrementalNIK(yumi_instance)
    tic = time.time()
    jnt_values_list = inik.gen_linear_motion(jlc_name, start_info, goal_info)
    toc = time.time()
    print(toc - tic)
    for jnt_values in jnt_values_list:
        yumi_instance.fk(jlc_name, jnt_values)
        yumi_meshmodel = yumi_instance.gen_meshmodel()
        yumi_meshmodel.attach_to(base)
    base.run()

