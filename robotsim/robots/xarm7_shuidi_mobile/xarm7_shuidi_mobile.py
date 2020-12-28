import os
import copy
import math
import numpy as np
import basis.robotmath as rm
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.xarm7.xarm7 as xa
import robotsim.grippers.xarm_gripper.xarm_gripper as xag
import robotsim.robots.robot_interface as ri


class XArm7YunjiMobile(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm7_yunji_mobile", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # agv
        self.agv = jl.JLChain(pos=pos,
                              rotmat=rotmat,
                              homeconf=np.zeros(0),
                              name='agv')  # TODO: change to 3-dof
        self.agv.jnts[1]['loc_pos'] = np.array([0, .0, .34231])
        self.agv.lnks[0]['name'] = 'agv'
        self.agv.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.agv.lnks[0]['meshfile'] = os.path.join(this_dir, 'meshes', 'shuidi_agv_meter.stl')
        self.agv.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.agv.reinitialize()
        # arm
        arm_homeconf = np.zeros(7)
        arm_homeconf[1] = -math.pi / 3
        arm_homeconf[3] = math.pi / 12
        arm_homeconf[5] = -math.pi / 12
        self.arm = xa.XArm7(pos=self.agv.jnts[-1]['gl_posq'],
                            rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name='arm', enable_cc=False)
        # gripper
        self.hnd = xag.XArmGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                   rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                   name='hnd', enable_cc=False)
        # collision detection
        if enable_cc:
            self.cc.add_cdlnks(self.agv, [0])
            self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
            self.cc.add_cdlnks(self.hnd.lft_outer, [0, 1, 2])
            self.cc.add_cdlnks(self.hnd.rgt_outer, [1, 2])
            activelist = [self.agv.lnks[0],
                          self.arm.lnks[0],
                          self.arm.lnks[1],
                          self.arm.lnks[2],
                          self.arm.lnks[3],
                          self.arm.lnks[4],
                          self.arm.lnks[5],
                          self.arm.lnks[6],
                          self.hnd.lft_outer.lnks[0],
                          self.hnd.lft_outer.lnks[1],
                          self.hnd.lft_outer.lnks[2],
                          self.hnd.rgt_outer.lnks[1],
                          self.hnd.rgt_outer.lnks[2]]
            self.cc.set_active_cdlnks(activelist)
            fromlist = [self.agv.lnks[0],
                        self.arm.lnks[0],
                        self.arm.lnks[1],
                        self.arm.lnks[2]]
            intolist = [self.arm.lnks[4],
                        self.arm.lnks[5],
                        self.arm.lnks[6],
                        self.hnd.lft_outer.lnks[0],
                        self.hnd.lft_outer.lnks[1],
                        self.hnd.lft_outer.lnks[2],
                        self.hnd.rgt_outer.lnks[1],
                        self.hnd.rgt_outer.lnks[2]]
            self.cc.set_cdpair(fromlist, intolist)
        # tool center point
        self.tcp_jlc = self.arm  # which jlc is the tcp located at?
        self.tcp_jlc.tcp_jntid = -1
        self.tcp_jlc.tcp_loc_pos = np.array([0, 0, .07])
        self.tcp_jlc.tcp_loc_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []

    @property
    def is_fk_updated(self):
        return self.hnd.is_fk_updated

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.agv.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.tcp_jlc.get_gl_pose(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fix_to(self, pos, rotmat):
        self.move_to(pos=pos, rotmat=rotmat)

    def fk(self, jnt_values, jlc_name='arm'):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param jlc_name: 'arm', 'agv', or 'agv_arm'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        if jlc_name == 'arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 7:
                raise ValueError("An 1x7 npdarray must be specified to move the arm!")
            self.arm.fk(jnt_values=jnt_values)
            self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        elif jlc_name == 'agv':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 3:
                raise ValueError("An 1x7 npdarray must be specified to move the agv!")
            self.pos = np.zeros(3)
            self.pos[:2] = jnt_values[:2]
            self.rotmat = rm.rotmat_from_axangle([0, 0, 1], jnt_values[2])
            self.agv.fix_to(self.pos, self.rotmat)
            self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
            self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        elif jlc_name == 'agv_arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 10:
                raise ValueError("An 1x7 npdarray must be specified to move both the agv and the arm!")
            self.pos = np.zeros(3)
            self.pos[:2] = jnt_values[:2]
            self.rotmat = rm.rotmat_from_axangle([0, 0, 1], jnt_values[2])
            self.agv.fix_to(self.pos, self.rotmat)
            self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq']) # TODO repeated
            self.arm.fk(jnt_values=jnt_values[3:10])
            self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
            # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.tcp_jlc.get_gl_pose(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def rand_conf(self, jlc_name):
        if jlc_name == 'arm':
            return self.arm.rand_conf()
        if jlc_name == 'agv':
            raise NotImplementedError
        if jlc_name == 'agv_arm':
            raise NotImplementedError

    def jaw_to(self, jawwidth):
        self.hnd.jaw_to(jawwidth)

    def hold(self, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if jawwidth is not None:
            self.hnd.jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.tcp_jlc.get_loc_pose(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.agv.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6]]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def release(self, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if jawwidth is not None:
            self.hnd.jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collisionmodel'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm7_shuidi_mobile_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.agv.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jntid=tcp_jntid,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(tcp_loc_pos=None,
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
                      name='xarm_shuidi_mobile_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.agv.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.arm.gen_meshmodel(tcp_jntid=tcp_jntid,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.hnd.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])

    gm.gen_frame().attach_to(base)
    xav = XArm7YunjiMobile(enable_cc=True)
    xav.fk(np.array([0, 0, 0, 0, 0, 0, math.pi, 0, -math.pi / 6, 0, 0]), jlc_name='agv_arm')
    xav.jaw_to(.08)
    xav_meshmodel = xav.gen_meshmodel()
    xav_meshmodel.attach_to(base)
    xav.show_cdprimit()
    xav.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = xav.is_collided()
    toc = time.time()
    print(result, toc - tic)

    # xav_cpy = xav.copy()
    # xav_cpy.move_to(pos=np.array([.5,.5,0]),rotmat=rm.rotmat_from_axangle([0,0,1],-math.pi/3))
    # xav_meshmodel = xav_cpy.gen_meshmodel()
    # xav_meshmodel.attach_to(base)
    # xav_cpy.show_cdprimit()
    # tic = time.time()
    # result = xav_cpy.is_collided(otherrobot_list=[xav])
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
