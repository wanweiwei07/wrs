import os
import numpy as np
from panda3d.core import NodePath, CollisionNode, CollisionBox, Point3
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class RobotiqHE(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name='rtq_he'):
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
        self.jaw_range = np.array([.0, .05])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"),
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
            userdef_cdprim_fn=self._base_cdprim)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.2, .2, .2, 1])
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, 0.11])
        # self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "finger1.stl"),
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=.005)
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, 1, 1])
        # the 2nd joint (right finger, -y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "finger2.stl"),
            cdmesh_type=self.cdmesh_type,
            cdprim_type=mcm.const.CDPrimType.AABB,
            ex_radius=.005)
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([1, .5, .5, 1])
        # reinitialize
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .14]) + coupling_offset_pos
        # collision detection
        # collisions
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0],
                                self.jlc.jnts[0].lnk,
                                self.jlc.jnts[1].lnk)

    @staticmethod
    def _base_cdprim(ex_radius=None):
        pdcnd = CollisionNode("rtq_he_base")
        collision_primitive_c0 = CollisionBox(Point3(0.0, 0.0, 0.1),
                                              x=.032 + ex_radius, y=.029 + ex_radius, z=.01 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(0.0, 0.0, 0.05),
                                              x=.02 + ex_radius, y=.02 + ex_radius, z=.03 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    def fix_to(self, pos, rotmat, jaw_width=None):
        self._pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self._pos
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
    gripper = RobotiqHE()
    # grippers = RobotiqHE(coupling_offset_pos=np.array([0, 0, 0.0331]),
    #                  coupling_offset_rotmat=rm.rotmat_from_axangle([1, 0, 0], math.pi / 6))
    gripper.change_jaw_width(.05)
    gripper.gen_meshmodel().attach_to(base)
    gripper.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    gripper.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    gripper.change_jaw_width(.0)
    mesh_model = gripper.gen_meshmodel()
    mesh_model.attach_to(base)
    mesh_model.show_cdprim()
    base.run()
