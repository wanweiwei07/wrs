import os
import math
import numpy as np
import basis.robotmath as rm
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.irb14050.irb14050 as ya
import robotsim.grippers.yumi_gripper.yumi_gripper as yg


class Yumi(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3)):
        this_dir, this_filename = os.path.split(__file__)
        self.pos = pos
        self.rotmat = rotmat
        # lft
        self.lft_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='agv')
        self.lft_body.jnts[1]['loc_pos'] = np.array([0.05355, -0.0725, 0.41492])
        self.lft_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-0.9795, -0.5682, -2.3155)  # left from robot view
        self.lft_body.lnks[0]['name'] = "yumi_lft_body"
        self.lft_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_body.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "body.stl")
        self.lft_body.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.lft_body.reinitialize()
        lft_arm_homeconf = np.zeros(7)
        self.lft_arm = ya.IRB14050(pos=self.lft_body.jnts[-1]['gl_posq'],
                                   rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                   homeconf=lft_arm_homeconf)
        self.lft_hnd = yg.YumiGripper(pos=self.lft_arm.jnts[-1]['gl_posq'],
                                      rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        # rgt
        self.rgt_body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='agv')
        self.rgt_body.jnts[1]['loc_pos'] = np.array([0.05355, 0.07250, 0.41492])
        self.rgt_body.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0.9781, -0.5716, 2.3180)  # left from robot view
        self.rgt_body.lnks[0]['name'] = "yumi_rgt_body"
        self.rgt_body.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_body.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.rgt_body.reinitialize()
        rgt_arm_homeconf = np.zeros(7)
        self.rgt_arm = ya.IRB14050(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                   rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                   homeconf=rgt_arm_homeconf)
        self.rgt_hnd = yg.YumiGripper(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                                      rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_body.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                            rotmat=self.lft_body.jnts[-1]['gl_rotmatq'])
        self.lft_hnd.fix_to(pos=self.lft_arm.jnts[-1]['gl_posq'],
                            rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'])
        self.rgt_hnd.fix_to(pos=self.rgt_arm.jnts[-1]['gl_posq'],
                            rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

    def fk(self, general_jnt_values, armname='lft'):
        """
        :param general_jnt_values: [nparray, nparray], 7+7, meter-radian
        :armname 'lft', 'rgt', 'both'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        # examine length
        if armname == 'lft' or armname == 'rgt':
            if not isinstance(general_jnt_values, np.ndarray) or general_jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move a single arm!")
            if armname == 'lft':
                self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                                    rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                    jnt_values=general_jnt_values)
            else:
                self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                    rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                    jnt_values=general_jnt_values)
        elif armname == 'both':
            if (not isinstance(general_jnt_values, list)
                    or general_jnt_values[0].size != 7
                    or general_jnt_values[1].size != 7):
                raise ValueError("A list of two 1x7 npdarrays must be specified to move both arm!")
            self.lft_arm.fix_to(pos=self.lft_body.jnts[-1]['gl_posq'],
                                rotmat=self.lft_body.jnts[-1]['gl_rotmatq'],
                                jnt_values=general_jnt_values[0])
            self.rgt_arm.fix_to(pos=self.rgt_body.jnts[-1]['gl_posq'],
                                rotmat=self.rgt_body.jnts[-1]['gl_rotmatq'],
                                jnt_values=general_jnt_values[1])

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggletcpcs=False,
                       togglejntscs=False,
                       toggleconnjnt=False,
                       name='xarm7_shuidi_mobile'):
        stickmodel = mc.ModelCollection(name=name)
        self.rgt_body.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggletcpcs=False,
                                     togglejntscs=togglejntscs).attach_to(stickmodel)
        self.lft_arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggletcpcs=toggletcpcs,
                                    togglejntscs=togglejntscs,
                                    toggleconnjnt=toggleconnjnt).attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggletcpcs=False,
                                    togglejntscs=togglejntscs,
                                    toggleconnjnt=toggleconnjnt).attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                    tcp_loc_pos=tcp_loc_pos,
                                    tcp_loc_rotmat=tcp_loc_rotmat,
                                    toggletcpcs=toggletcpcs,
                                    togglejntscs=togglejntscs,
                                    toggleconnjnt=toggleconnjnt).attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggletcpcs=False,
                                    togglejntscs=togglejntscs,
                                    toggleconnjnt=toggleconnjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggletcpcs=False,
                      togglejntscs=False,
                      name='xarm_gripper_meshmodel', rgba=None):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_body.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggletcpcs=False,
                                    togglejntscs=togglejntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft_arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggletcpcs=toggletcpcs,
                                   togglejntscs=togglejntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel(tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   toggletcpcs=False,
                                   togglejntscs=togglejntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                                   tcp_loc_pos=tcp_loc_pos,
                                   tcp_loc_rotmat=tcp_loc_rotmat,
                                   toggletcpcs=toggletcpcs,
                                   togglejntscs=togglejntscs,
                                   rgba=rgba).attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel(tcp_loc_pos=None,
                                   tcp_loc_rotmat=None,
                                   toggletcpcs=False,
                                   togglejntscs=togglejntscs,
                                   rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])
    gm.gen_frame().attach_to(base)
    yumi_instance = Yumi()

    yumi_meshmodel = yumi_instance.gen_meshmodel()
    yumi_meshmodel.attach_to(base)
    yumi_meshmodel.show_cdprimit()
    yumi_instance.gen_stickmodel().attach_to(base)
    base.run()
