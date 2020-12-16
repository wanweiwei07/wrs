import os
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
        self.agv = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='agv')  # TODO: change to 3-dof
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
        self.arm = xa.XArm7(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'],
                            homeconf=arm_homeconf, name=self.name + "_arm")
        self.arm.disable_localcc()
        # gripper
        self.hnd = xag.XArmGripper(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                   name=self.name + "_hnd")
        # objects in hand
        self.obj_inhnd_cdprimit_cache = []
        # collision detection
        self.cc = cc.CollisionChecker(self.name + "_collisionchecker")
        self.cc.add_cdlnks(self.agv, [0])
        self.cc.add_cdlnks(self.arm, [0,1,2,3,4,5,6])
        self.cc.add_cdlnks(self.hnd.lft_outer, [0,1,2])
        self.cc.add_cdlnks(self.hnd.rgt_outer, [1,2])
        activelist = [self.agv.lnks[0]['cdprimit_cache'],
                      self.arm.lnks[0]['cdprimit_cache'],
                      self.arm.lnks[1]['cdprimit_cache'],
                      self.arm.lnks[2]['cdprimit_cache'],
                      self.arm.lnks[3]['cdprimit_cache'],
                      self.arm.lnks[4]['cdprimit_cache'],
                      self.arm.lnks[5]['cdprimit_cache'],
                      self.arm.lnks[4]['cdprimit_cache'],
                      self.arm.lnks[6]['cdprimit_cache'],
                      self.hnd.lft_outer.lnks[0]['cdprimit_cache'],
                      self.hnd.lft_outer.lnks[1]['cdprimit_cache'],
                      self.hnd.lft_outer.lnks[2]['cdprimit_cache'],
                      self.hnd.rgt_outer.lnks[1]['cdprimit_cache'],
                      self.hnd.rgt_outer.lnks[2]['cdprimit_cache']]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.agv.lnks[0]['cdprimit_cache'],
                    self.arm.lnks[0]['cdprimit_cache'],
                    self.arm.lnks[1]['cdprimit_cache'],
                    self.arm.lnks[2]['cdprimit_cache']]
        intolist = [self.arm.lnks[5]['cdprimit_cache'],
                    self.arm.lnks[6]['cdprimit_cache'],
                    self.hnd.lft_outer.lnks[0]['cdprimit_cache'],
                    self.hnd.lft_outer.lnks[1]['cdprimit_cache'],
                    self.hnd.lft_outer.lnks[2]['cdprimit_cache'],
                    self.hnd.rgt_outer.lnks[1]['cdprimit_cache'],
                    self.hnd.rgt_outer.lnks[2]['cdprimit_cache']]
        self.cc.set_cdpair(fromlist, intolist)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.objs_inhnd_infos = []

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.agv.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.agv.jnts[-1]['gl_posq'], rotmat=self.agv.jnts[-1]['gl_rotmatq'])
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # set objects in hand cache's need-to-update marker to True
        for one_oih_info in self.objs_inhnd_infos:
            gl_pos, gl_rotmat = self.hnd.lft_outer.get_worldpose(one_oih_info['rel_pos'], one_oih_info['rel_rotmat'])
            one_oih_info['gl_pos'] = gl_pos
            one_oih_info['gl_rotmat'] = gl_rotmat
            one_oih_info['cdprimit_cache'][0] = True

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
        for one_oih_info in self.objs_inhnd_infos:
            gl_pos, gl_rotmat = self.hnd.lft_outer.get_worldpose(one_oih_info['rel_pos'], one_oih_info['rel_rotmat'])
            one_oih_info['gl_pos'] = gl_pos
            one_oih_info['gl_rotmat'] = gl_rotmat
            one_oih_info['cdprimit_cache'][0] = True

    def hold(self, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if jawwidth is not None:
            self.hnd.jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.hnd.lft_outer.get_relpose(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.agv.lnks[0]['cdprimit_cache'],
                    self.arm.lnks[0]['cdprimit_cache'],
                    self.arm.lnks[1]['cdprimit_cache'],
                    self.arm.lnks[2]['cdprimit_cache'],
                    self.arm.lnks[3]['cdprimit_cache'],
                    self.arm.lnks[4]['cdprimit_cache'],
                    self.arm.lnks[5]['cdprimit_cache'],
                    self.arm.lnks[6]['cdprimit_cache']]
        self.objs_inhnd_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))

    def release(self, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if jawwidth is not None:
            self.hnd.jaw_to(jawwidth)
        for one_oih_info in self.objs_inhnd_infos:
            if one_oih_info['collisionmodel'] is objcm:
                self.cc.delete_cdobj(one_oih_info)
                self.objs_inhnd_infos.remove(one_oih_info)
                break

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        return self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)

    def gen_stickmodel(self, name='xarm7_shuidi_mobile'):
        stickmodel = mc.ModelCollection(name=name)
        self.agv.gen_stickmodel().attach_to(stickmodel)
        self.arm.gen_stickmodel().attach_to(stickmodel)
        self.hnd.gen_stickmodel().attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.agv.gen_meshmodel().attach_to(meshmodel)
        self.arm.gen_meshmodel().attach_to(meshmodel)
        self.hnd.gen_meshmodel().attach_to(meshmodel)
        for obj_info in self.objs_inhnd_infos:
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
    xav = XArm7YunjiMobile()
    xav.fk(np.array([0,0,0,0,math.pi *2/ 3,0,math.pi,0,-math.pi / 6,0,0]))
    xav_meshmodel = xav.gen_meshmodel()
    xav_meshmodel.attach_to(base)
    xav_meshmodel.show_cdprimit()
    xav.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = xav.is_collided()
    toc = time.time()
    print(result, toc-tic)
    base.run()
