import os
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi


class WRSTweezer(gpi.GripperInterface):
    """
    This class is the second version of Junbo's extension grippers. It is designed to be used with Denso Cobotta
    author: weiwei
    date: 20240909
    """

    def __init__(self, pos=rm.zeros(3), rotmat=rm.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name="wrs_tweezer"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = rm.vec(0.0, .004)
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "tweezer_palm.stl"),
            name=name + "_palm_cmodel",
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=rm.vec(0, self.jaw_range[1] / 2))
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "tweezer_pin.stl"),
                                                         name=name + "_pin1_cmodel",
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.AABB, ex_radius=.002)
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.np.array([1, .5, .5, 1])
        # the 2nd joint (left finger, +y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=rm.vec(0.0, self.jaw_range[1]))
        self.jlc.jnts[1].loc_pos = rm.vec(0, -.001, 0)
        self.jlc.jnts[1].loc_motion_ax = -rm.const.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "tweezer_pin.stl"),
                                                         name=name + "_pin2_cmodel",
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.002)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.np.array([.5, .5, 1, 1])
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = rm.vec(-0.024, 0, .122)
        # collision detection
        # collisions
        self.cdelements = (self.jlc.anchor.lnk_list[0],
                           self.jlc.jnts[0].lnk,
                           self.jlc.jnts[1].lnk)

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
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, side_jawwidth*2])
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name + '_stickmodel')
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name + '_meshmodel')
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

    base = wd.World(cam_pos=[-.5, .0, .08], lookat_pos=[0, 0, 0.08], auto_rotate=False)
    mcm.mgm.gen_frame().attach_to(base)
    gripper = WRSTweezer()
    gripper.change_jaw_width(0.005)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=False).attach_to(base)

    base.run()
