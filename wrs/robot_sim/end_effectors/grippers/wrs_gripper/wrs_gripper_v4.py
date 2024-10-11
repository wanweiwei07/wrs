import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class WRSGripper4(gpi.GripperInterface):
    """
    This class is the second version of Junbo's extension grippers. It is designed to be used with Denso Cobotta
    author: weiwei
    date: 20240909
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name="wrs_gripper4"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = np.array([0.0, .1344])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=6, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r11.stl"),
                                                                cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        # rack length
        self._rack_length = 0.0224
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 3]))
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r21.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._mid_cdprim, ex_radius=.002)
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.tab20_list[5]
        # the 2nd joint (left finger, +y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r12.stl"),
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.002)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.tab20_list[1]
        # the 3nd joint (left finger, +y direction)
        self.jlc.jnts[2].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[2].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r22.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._fgr_cdprim, ex_radius=.002)
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.tab20_list[3]
        # the 1st joint (rgt finger, -y direction)
        self.jlc.jnts[3].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 3]))
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(.0, .0, np.pi)
        self.jlc.jnts[3].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r21.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._mid_cdprim, ex_radius=.002)
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.tab20_list[5]
        # the 2nd joint (rgt finger, -y direction)
        self.jlc.jnts[4].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[4].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r12.stl"),
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.002)
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.tab20_list[1]
        # the 3nd joint (rgt finger, -y direction)
        self.jlc.jnts[5].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[5].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v4_r22.stl"),
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._fgr_cdprim, ex_radius=.002)
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.tab20_list[3]
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .15])
        # collision detection
        # collisions
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0],
                                self.jlc.jnts[3].lnk,
                                self.jlc.jnts[5].lnk)

    @staticmethod
    def _fgr_cdprim(ex_radius):
        pdcnd = CollisionNode("finger")
        collision_primitive_c0 = CollisionBox(Point3(.0, 0.0224*3+0.002, .102),
                                              x=.0075 + ex_radius, y=0.001 + ex_radius, z=.06 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.013, 0.0224*3+0.002, .069),
                                              x=.002 + ex_radius, y=0.001 + ex_radius, z=.005 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        collision_primitive_c2 = CollisionBox(Point3(-.02, 0.0224*3-0.015, .052),
                                              x=.015 + ex_radius, y=0.014 + ex_radius, z=.012 + ex_radius)
        pdcnd.addSolid(collision_primitive_c2)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _mid_cdprim(ex_radius):
        pdcnd = CollisionNode("mid_plate")
        collision_primitive_c0 = CollisionBox(Point3(-.011 - .02 / 2, .025 - 0.03 / 2, .003 + .06 / 2),
                                              x=.02 / 2 + ex_radius, y=0.03 / 2 + ex_radius, z=.06 / 2 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        cdprim = NodePath("user_defined")
        cdprim.attachNewNode(pdcnd)
        return cdprim

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
            motion_rack_length = self._rack_length - side_jawwidth / 3.0
            self.jlc.goto_given_conf(jnt_values=[-motion_rack_length, -motion_rack_length, -motion_rack_length,
                                                 -4*motion_rack_length, -motion_rack_length, -motion_rack_length])
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='wg3_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False, name='wg3_meshmodel'):
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
        self._gen_oiee_meshmodel(m_col=m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                 toggle_cdmesh=toggle_cdmesh)
        return m_col


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[-.5, .0, .08], lookat_pos=[0, 0, 0.08], auto_cam_rotate=False)
    mcm.mgm.gen_frame().attach_to(base)
    gripper = WRSGripper4()
    gripper.change_jaw_width(.1344)
    # grippers.change_jaw_width(0)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=False).attach_to(base)

    base.run()