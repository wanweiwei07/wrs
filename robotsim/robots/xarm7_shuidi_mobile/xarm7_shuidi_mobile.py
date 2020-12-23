import os
import copy
import math
import numpy as np
import basis.robotmath as rm
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.xarm7.xarm7 as xa
import robotsim.grippers.xarm_gripper.xarm_gripper as xag
import robotsim._kinematics.collisionchecker as cc


class XArm7YunjiMobile(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm7_yunji_mobile"):
        self.name = name
        this_dir, this_filename = os.path.split(__file__)
        # agv
        self.pos = pos
        self.rotmat = rotmat
        self.agv = jl.JLChain(pos=pos,
                              rotmat=rotmat,
                              homeconf=np.zeros(0),
                              name=self.name + '_agv')  # TODO: change to 3-dof
        self.agv.jnts[1]['loc_pos'] = np.array([0, .0, .34231])
        self.agv.lnks[0]['name'] = self.name + "_agv"
        self.agv.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.agv.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "shuidi_agv_meter.stl")
        self.agv.lnks[0]['rgba'] = [.35, .35, .35, 1.0]
        self.agv.reinitialize()
        self.agv.disable_localcc()
        # arm
        arm_homeconf = np.zeros(7)
        arm_homeconf[1] = -math.pi / 3
        arm_homeconf[3] = math.pi / 12
        arm_homeconf[5] = -math.pi / 12
        self.arm = xa.XArm7(pos=self.agv.jnts[-1]['gl_posq'],
                            rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf,
                            name=self.name + "_arm")
        self.arm.disable_localcc()
        # gripper
        self.hnd = xag.XArmGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                   rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                   name=self.name + "_hnd")
        # collision detection
        self.cc = self._setup_collisionchecker()
        # tool center point
        self.tcp_jlc = self.arm  # which jlc is the tcp located at?
        self.tcp_jlc.tcp_jntid = -1
        self.tcp_jlc.tcp_loc_pos = np.array([0, 0, .07])
        self.tcp_jlc.tcp_loc_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.objs_inhnd_infos = []

    def _setup_collisionchecker(self):
        checker = cc.CollisionChecker("collision_checker")
        checker.add_cdlnks(self.agv, [0])
        checker.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        checker.add_cdlnks(self.hnd.lft_outer, [0, 1, 2])
        checker.add_cdlnks(self.hnd.rgt_outer, [1, 2])
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
        checker.set_active_cdlnks(activelist)
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
        checker.set_cdpair(fromlist, intolist)
        return checker

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.agv.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # set objects in hand cache's need-to-update marker to True
        for one_oih_info in self.objs_inhnd_infos:
            gl_pos, gl_rotmat = self.tcp_jlc.get_gl_pose(one_oih_info['rel_pos'], one_oih_info['rel_rotmat'])
            one_oih_info['gl_pos'] = gl_pos
            one_oih_info['gl_rotmat'] = gl_rotmat

    def fk(self, general_jnt_values):
        """
        :param general_jnt_values: 3+7 or 3+7+1, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.pos = np.zeros(3)
        self.pos[:2] = general_jnt_values[:2]
        self.rotmat = rm.rotmat_from_axangle([0, 0, 1], general_jnt_values[2])
        self.agv.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                        jnt_values=general_jnt_values[3:10])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        if len(general_jnt_values) == 11:  # gripper is also set
            self.hnd.jaw_to(general_jnt_values[10])
        # set objects in hand cache's need-to-update marker to True
        for obj_info in self.objs_inhnd_infos:
            gl_pos, gl_rotmat = self.tcp_jlc.get_gl_pose(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

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
        self.objs_inhnd_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))

    def get_hold_objlist(self):
        return_list = []
        for obj_info in self.objs_inhnd_infos:
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
        for obj_info in self.objs_inhnd_infos:
            if obj_info['collisionmodel'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.objs_inhnd_infos.remove(obj_info)
                break

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        # object in hand do not update by itself
        is_fk_updated = self.agv.is_fk_updated or self.arm.is_fk_updated or self.hnd.lft_outer.is_fk_updated
        return self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list,
                                   need_update=is_fk_updated)

    def show_cdprimit(self):
        is_fk_updated = self.agv.is_fk_updated or self.arm.is_fk_updated or self.hnd.lft_outer.is_fk_updated
        self.cc.show_cdprimit(need_update=is_fk_updated)

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
        for obj_info in self.objs_inhnd_infos:
            objcm = obj_info['collisionmodel']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.attach_to(meshmodel)
        return meshmodel

    def copy(self):
        self_copy = copy.deepcopy(self)
        # update colliders; they are problematic, I have to update it manually
        for child in self_copy.cc.np.getChildren():
            self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[1.5, 0, 3], lookatpos=[0, 0, .5])

    gm.gen_frame().attach_to(base)
    xav = XArm7YunjiMobile()
    xav.fk(np.array([0, 0, 0, 0, 0, 0, math.pi, 0, -math.pi / 6, 0, 0]))
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
