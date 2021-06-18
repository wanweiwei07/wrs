import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.robots.robot_interface as ri


class TBM(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="tbm"):
        self.pos = pos
        self.rotmat = rotmat
        self.name = name
        this_dir, this_filename = os.path.split(__file__)
        # agv
        self.front = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(2), name='head')
        self.front.jnts[1]['loc_pos'] = np.zeros(3)
        self.front.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.front.lnks[0]['name'] = 'tbm_front_shield'
        self.front.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.front.lnks[0]['meshfile'] = os.path.join(this_dir, 'meshes', 'tbm_front_shield.stl')
        self.front.lnks[0]['rgba'] = [.7, .7, .7, 1.0]
        self.front.lnks[1]['name'] = 'tbm_cutter_head'
        self.front.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.front.lnks[1]['meshfile'] = os.path.join(this_dir, 'meshes', 'tbm_cutter_head.stl')
        self.front.lnks[1]['rgba'] = [.35, .35, .35, 1.0]
        self.front.tgtjnts = [1]
        self.front.reinitialize()

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.front.fix_to(self.pos, self.rotmat)

    def fk(self, jnt_values=np.zeros(1)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.front.fk(jnt_values=jnt_values)

    def get_jnt_values(self):
        return self.front.get_jnt_values()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='tbm'):
        stickmodel = mc.ModelCollection(name=name)
        self.front.gen_stickmodel(tcp_loc_pos=None,
                                  tcp_loc_rotmat=None,
                                  toggle_tcpcs=False,
                                  toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='tbm'):
        meshmodel = mc.ModelCollection(name=name)
        self.front.gen_meshmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs,
                                 rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[50, 0, 10], lookat_pos=[0, 0, 0])

    gm.gen_frame().attach_to(base)
    otherbot_s = TBM()
    otherbot_s.fk(jnt_values=np.array([0]))
    otherbot_s.gen_meshmodel().attach_to(base)
    base.run()
