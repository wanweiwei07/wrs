import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc


class OR2FG7(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name='or2fg7'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        self.coupling.loc_flange_pose_list[0] = (coupling_offset_pos, coupling_offset_rotmat)
        if np.any(self.coupling.loc_flange_pose_list[0][0]):
            self.coupling.lnk_list[0].cmodel = mcm.gen_stick(spos=np.zeros(3),
                                                             epos=self.coupling.loc_flange_pose_list[0][0],
                                                             type="rect",
                                                             radius=0.035,
                                                             rgb=np.array([.35, .35, .35]),
                                                             alpha=1,
                                                             n_sec=24)
        # jaw range
        self.jaw_range = np.array([.0, .038])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base_link.stl"),
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[14]
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([0.0, -0.019, 0.0])
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "inward_left_finger_link.stl"),
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._finger_cdprim, ex_radius=.005)
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, 1, 1])
        # the 2nd joint (right finger, -y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([0.0, 0.038, 0.0])
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "inward_right_finger_link.stl"),
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._finger_cdprim, ex_radius=.005)
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([1, .5, .5, 1])
        # reinitialize
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .15]) + coupling_offset_pos
        # collision detection
        # collisions
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0],
                                self.jlc.jnts[0].lnk,
                                self.jlc.jnts[1].lnk)

    @staticmethod
    def _finger_cdprim(ex_radius):
        pdcnd = CollisionNode("finger")
        collision_primitive_c0 = CollisionBox(Point3(.015, .012, .08),
                                              x=.0035 + ex_radius, y=0.0032 + ex_radius, z=.05 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(.008, .0, .008),
                                              x=.018 + ex_radius, y=0.011 + ex_radius, z=.011 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self.pos
        self.coupling.rotmat = self.rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def get_jaw_width(self):
        return self.jlc.jnts[1].motion_value

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if self.jaw_range[0] / 2 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, -jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='or2fg7_stickmodel'):
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

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mcm.mgm.gen_frame().attach_to(base)
    gripper = OR2FG7(coupling_offset_pos=np.array([0, 0, 0.0145]))
    gripper.change_jaw_width(.038)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True, toggle_jnt_frames=True).attach_to(base)
    gripper.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    gripper2 = OR2FG7(pos=np.array([0.0, .5, 0.0]))
    gripper2.change_jaw_width(.001)
    gripper2.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True, toggle_jnt_frames=True).attach_to(base)
    base.run()