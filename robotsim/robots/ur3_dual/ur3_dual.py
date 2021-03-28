import os
import math
import numpy as np
import basis.robot_math as rm
import modeling.modelcollection as mc
import modeling.collisionmodel as cm
import robotsim._kinematics.jlchain as jl
import robotsim.manipulators.ur3.ur3 as ur
import robotsim.grippers.robotiq85.robotiq85 as rtq
from panda3d.core import CollisionNode, CollisionBox, Point3
import robotsim.robots.robot_interface as ri


class UR3Dual(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='ur3dual', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.lft_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='lft_base_jl')
        self.lft_base.jnts[1]['loc_pos'] = np.array([.0, .258485281374, 1.61051471863])  # left from robot view
        self.lft_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-3.0 * math.pi / 4.0, 0, math.pi, 'rxyz')  # left from robot view
        self.lft_base.lnks[0]['name'] = "ur3_dual_lft_base"
        self.lft_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.lft_base.lnks[0]['collisionmodel'] = cm.CollisionModel(
            os.path.join(this_dir, "meshes", "ur3_dual_base.stl"),
            cdprimit_type="user_defined", expand_radius=.005,
            userdefined_cdprimitive_fn=self._base_combined_cdnp)
        self.lft_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.lft_base.reinitialize()
        lft_arm_homeconf = np.zeros(6)
        lft_arm_homeconf[0] = math.pi / 3.0
        lft_arm_homeconf[1] = -math.pi * 1.0 / 3.0
        lft_arm_homeconf[2] = -math.pi * 2.0 / 3.0
        lft_arm_homeconf[3] = math.pi
        lft_arm_homeconf[4] = -math.pi / 2.0
        self.lft_arm = ur.UR3(pos=self.lft_base.jnts[-1]['gl_posq'],
                              rotmat=self.lft_base.jnts[-1]['gl_rotmatq'],
                              homeconf=lft_arm_homeconf,
                              enable_cc=False)
        # lft hand offset (if needed)
        self.lft_hnd_offset = np.array([0, 0, 0.0484])
        lft_hnd_pos, lft_hnd_rotmat = self.lft_arm.cvt_loc_intcp_to_gl(loc_pos=self.lft_hnd_offset)
        self.lft_hnd = rtq.Robotiq85(pos=lft_hnd_pos,
                                     rotmat=self.lft_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # rigth side
        self.rgt_base = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='rgt_base_jl')
        self.rgt_base.jnts[1]['loc_pos'] = np.array([.0, -.258485281374, 1.61051471863])  # right from robot view
        self.rgt_base.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(3.0 * math.pi / 4.0, .0, .0)  # left from robot view
        self.rgt_base.lnks[0]['name'] = "ur3_dual_rgt_base"
        self.rgt_base.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.rgt_base.lnks[0]['meshfile'] = None
        self.rgt_base.lnks[0]['rgba'] = [.3, .3, .3, 1.0]
        self.rgt_base.reinitialize()
        rgt_arm_homeconf = np.zeros(6)
        rgt_arm_homeconf[0] = -math.pi * 1.0 / 3.0
        rgt_arm_homeconf[1] = -math.pi * 2.0 / 3.0
        rgt_arm_homeconf[2] = math.pi * 2.0 / 3.0
        rgt_arm_homeconf[4] = math.pi / 2.0
        self.rgt_arm = ur.UR3(pos=self.rgt_base.jnts[-1]['gl_posq'],
                              rotmat=self.rgt_base.jnts[-1]['gl_rotmatq'],
                              homeconf=rgt_arm_homeconf,
                              enable_cc=False)
        # rgt hand offset (if needed)
        self.rgt_hnd_offset = np.array([0, 0, 0.0484])
        rgt_hnd_pos, rgt_hnd_rotmat = self.rgt_arm.cvt_loc_intcp_to_gl(loc_pos=self.rgt_hnd_offset)
        # TODO replace using copy
        self.rgt_hnd = rtq.Robotiq85(pos=rgt_hnd_pos,
                                     rotmat=self.rgt_arm.jnts[-1]['gl_rotmatq'],
                                     enable_cc=False)
        # tool center point
        # lft
        self.lft_arm.tcp_jntid = -1
        self.lft_arm.tcp_loc_pos = np.array([0, 0, .145])
        self.lft_arm.tcp_loc_rotmat = np.eye(3)
        # rgt
        self.rgt_arm.tcp_jntid = -1
        self.rgt_arm.tcp_loc_pos = np.array([0, 0, .145])
        self.rgt_arm.tcp_loc_rotmat = np.eye(3)
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.lft_oih_infos = []
        self.rgt_oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()

    @staticmethod
    def _base_combined_cdnp(name, radius):
        collision_node = CollisionNode(name)
        collision_primitive_c0 = CollisionBox(Point3(0.18, 0.0, 0.105),
                                              x=.61 + radius, y=.41 + radius, z=.105 + radius)
        collision_node.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, 0.4445),
                                              x=.321 + radius, y=.321 + radius, z=.2345 + radius)
        collision_node.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(0.0, 0.0, 0.8895),
                                              x=.05 + radius, y=.05 + radius, z=.6795 + radius)
        collision_node.addSolid(collision_primitive_c2)
        collision_primitive_c3 = CollisionBox(Point3(0.0, 0.0, 1.619),
                                              x=.1 + radius, y=.275 + radius, z=.05 + radius)
        collision_node.addSolid(collision_primitive_c3)
        collision_primitive_l0 = CollisionBox(Point3(0.0, 0.300, 1.669),
                                              x=.1 + radius, y=.029 + radius, z=.021 + radius)
        collision_node.addSolid(collision_primitive_l0)
        collision_primitive_r0 = CollisionBox(Point3(0.0, -0.300, 1.669),
                                              x=.1 + radius, y=.029 + radius, z=.021 + radius)
        collision_node.addSolid(collision_primitive_r0)
        return collision_node

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError

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
    u3d = UR3Dual()
    # u3d.fk(.85)
    u3d_meshmodel = u3d.gen_meshmodel()
    u3d_meshmodel.attach_to(base)
    u3d_meshmodel.show_cdprimit()
    u3d.gen_stickmodel().attach_to(base)
    base.run()
