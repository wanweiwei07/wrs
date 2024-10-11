import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class WRSGripper2(gpi.GripperInterface):
    """
    This class is the second version of the Lite6 WRS grippers. It is designed to be used with the xArm Lite 6 robot.
    Hao developed the original code for this grippers.
    Weiwei kept it updated.
    author: hao, weiwei
    date: 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name="wrs_gripper2"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = np.array([0.0, .08])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "base_v2.stl"),
                                                                cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[14]
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([-0.01492498,- 0.005, .05])
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "finger_v2.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._finger_cdprim, ex_radius=.005)
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.tab20_list[14]
        # the 2nd joint (right finger, -y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([0.02984996, 0.01, .0])
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi)
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "finger_v2.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._finger_cdprim, ex_radius=.005)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.tab20_list[14]
        # reinitialize
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .175])
        # collision detection
        # collisions
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0],
                                self.jlc.jnts[0].lnk,
                                self.jlc.jnts[1].lnk)
        # update jaw width
        self.change_jaw_width(jaw_width=self.jaw_width)

    @staticmethod
    def _finger_cdprim(ex_radius):
        pdcnd = CollisionNode("finger")
        collision_primitive_c0 = CollisionBox(Point3(.015,.012,.08),
                                              x=.0035 + ex_radius, y=0.0032 + ex_radius, z=.05 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.008, .0, .008),
                                              x=.018 + ex_radius, y=0.011 + ex_radius, z=.011 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    # @staticmethod
    # def _hnd_base_cdnp(name, radius):
    #     collision_node = CollisionNode(name)
    #     collision_primitive_c0 = CollisionBox(Point3(0, 0, .031),
    #                                           x=.036 + radius, y=0.038 + radius, z=.031 + radius)
    #     collision_node.addSolid(collision_primitive_c0)  # 0.62
    #     collision_primitive_c1 = CollisionBox(Point3(0, 0, .067),
    #                                           x=.036 + radius, y=0.027 + radius, z=.003 + radius)
    #     collision_node.addSolid(collision_primitive_c1)  # 0.06700000
    #     #
    #     collision_primitive_c2 = CollisionBox(Point3(.006, .049, .0485),
    #                                           x=.02 + radius, y=.02 + radius, z=.015 + radius)
    #     collision_node.addSolid(collision_primitive_c2)
    #     collision_primitive_c3 = CollisionBox(Point3(0, 0, .08),
    #                                           x=.013 + radius, y=0.013 + radius, z=.005 + radius)
    #     collision_node.addSolid(collision_primitive_c3)
    #
    #     return collision_node

    def fix_to(self, pos, rotmat, jaw_width=None):
        self._pos = pos
        self._rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self._pos
        self.coupling.rotmat = self._rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def get_jaw_width(self):
        return self.jlc.jnts[1].motion_value

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if self.jaw_range[0] / 2 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, jaw_width])
            self.jaw_width = jaw_width
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='lite6_wrs_gripper_v2_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False, name='lite6_wrs_gripper_v2_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_flange_frame=False,
                                    toggle_root_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.jlc.gen_meshmodel(rgb=rgb,
                               alpha=alpha,
                               toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self._gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_cam_rotate=False)
    mgm.gen_frame().attach_to(base)
    gripper = WRSGripper2()
    gripper.change_jaw_width(.04)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    # grippers.show_cdprimit()
    base.run()
