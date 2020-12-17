import os
import math
import numpy as np
import basis.robotmath as rm
import modeling.modelcollection as mc
import modeling.collisionmodel as cm
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.ur3e.ur3e as ur
import robotsim.grippers.robotiqhe.robotiqhe as rtq


class CollisionModel_User(cm.CollisionModel):
    def __init__(self, objinit, btransparency=True, expand_radius=None, name="defaultname"):
        super().__init__(objinit, btransparency=btransparency, cdprimitive_type="userdefined",
                         expand_radius=expand_radius, name=name)




class UR3EDual(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3)):
        this_dir, this_filename = os.path.split(__file__)
        self.pos = pos
        self.rotmat = rotmat
        # left side TODO: direct cm, other than base link
        self.lft_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='agv')
        self.lft_base.jnts[1]['loc_pos'] = np.array([.365, .345, 1.33])  # left from robot view
        self.lft_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot view
        self.lft_base.lnks[0]['name'] = "ur3e_dual_base"
        self.lft_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        # manually specify cd primitives
        self.lft_base.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "ur3e_dual_base.stl")
        self.lft_base.lnks[0]['collisionmodel'] = cm.CollisionModel(self.lft_base.lnks[0]['meshfile'],
                                                                    cdprimitive_type="external",
                                                                    userdefined_cdprimitive_callback=self._base_primitive)
        self.lft_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.lft_base.reinitialize()
        lft_arm_homeconf = np.zeros(6)
        lft_arm_homeconf[0] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[2] = math.pi / 2.0
        lft_arm_homeconf[3] = math.pi
        lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = ur.UR3E(pos=self.lft_base.jnts[-1]['gl_posq'], rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                               homeconf=lft_arm_homeconf)
        # lft hand offset (if needed)
        self.lft_hnd_offset = np.zeros(3)
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.lft_hnd_offset)
        self.lft_hnd = rtq.RobotiqHE(pos=lft_hnd_pos, rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'])
        # rigth side
        self.rgt_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='agv')
        self.rgt_base.jnts[1]['loc_pos'] = np.array([.365, -.345, 1.33])  # right from robot view
        self.rgt_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(math.pi / 2.0, 0,
                                                                   math.pi / 2.0)  # left from robot view
        self.rgt_base.lnks[0]['name'] = "ur3e_dual_base"
        self.rgt_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_base.lnks[0]['meshfile'] = None
        self.rgt_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.rgt_base.reinitialize()
        rgt_arm_homeconf = np.zeros(6)
        rgt_arm_homeconf[0] = math.pi * 2.0 / 3.0
        rgt_arm_homeconf[1] = -math.pi / 3.0
        rgt_arm_homeconf[2] = -math.pi / 2.0
        rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = ur.UR3E(pos=self.rgt_base.jnts[-1]['gl_posq'], rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                               homeconf=rgt_arm_homeconf)
        # rgt hand offset (if needed)
        self.rgt_hnd_offset = np.zeros(3)
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        # TODO replace using copy
        self.rgt_hnd = rtq.RobotiqHE(pos=rgt_hnd_pos, rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'])

    @staticmethod
    def _base_primitive(self, objcm, name, radius):
        collision_node = CollisionNode(name)
        collision_primitive = CollisionBox(bottom_left, top_right)

    def move_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.lft_base.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_base.jnts[-1]['gl_posq'], rotmat=self.lft_base.jnts[-1]['gl_rotmatq'])
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat)
        self.rgt_base.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_base.jnts[-1]['gl_posq'], rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'])
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat)

    def fk(self, general_jnt_values):
        """
        :param general_jnt_values: 3+7 or 3+7+1, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.pos = np.zeros(0)  # no translation in z
        self.pos[:2] = general_jnt_values[:2]
        self.rotmat = rm.rotmat_from_axangle([0, 0, 1], general_jnt_values[2])
        # left side
        self.lft_base.fix_to(self.pos, self.rotmat)
        self.lft_arm.fix_to(pos=self.lft_base.jnts[-1]['gl_posq'], rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                            jnt_values=general_jnt_values[3:10])
        if len(general_jnt_values) != 10:  # gripper is also set
            general_jnt_values[10] = None
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.lft_hnd.fix_to(pos=lft_hnd_pos, rotmat=lft_hnd_rotmat, jnt_values=general_jnt_values[10])
        # right side
        self.rgt_base.fix_to(self.pos, self.rotmat)
        self.rgt_arm.fix_to(pos=self.rgt_base.jnts[-1]['gl_posq'], rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                            jnt_values=general_jnt_values[3:10])
        if len(general_jnt_values) != 10:  # gripper is also set
            general_jnt_values[10] = None
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.get_worldpose(relpos=self.rgt_hnd_offset)
        self.rgt_hnd.fix_to(pos=rgt_hnd_pos, rotmat=rgt_hnd_rotmat, jnt_values=general_jnt_values[10])

    def gen_stickmodel(self, name='xarm7_shuidi_mobile'):
        stickmodel = mc.ModelCollection(name=name)
        self.lft_base.gen_stickmodel().attach_to(stickmodel)
        self.lft_arm.gen_stickmodel().attach_to(stickmodel)
        self.lft_hnd.gen_stickmodel().attach_to(stickmodel)
        self.rgt_base.gen_stickmodel().attach_to(stickmodel)
        self.rgt_arm.gen_stickmodel().attach_to(stickmodel)
        self.rgt_hnd.gen_stickmodel().attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.lft_base.gen_meshmodel().attach_to(meshmodel)
        self.lft_arm.gen_meshmodel().attach_to(meshmodel)
        self.lft_hnd.gen_meshmodel().attach_to(meshmodel)
        self.rgt_base.gen_meshmodel().attach_to(meshmodel)
        self.rgt_arm.gen_meshmodel().attach_to(meshmodel)
        self.rgt_hnd.gen_meshmodel().attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[3, 0, 3], lookatpos=[0, 0, 1])
    gm.gen_frame().attach_to(base)
    u3ed = UR3EDual()
    # u3ed.fk(.85)
    u3ed_meshmodel = u3ed.gen_meshmodel()
    u3ed_meshmodel.attach_to(base)
    u3ed_meshmodel.show_cdprimit()
    u3ed.gen_stickmodel().attach_to(base)
    base.run()
