import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
from wrs.motion.primitives.approach_depart_planner import ADPlanner


class A2WGripper(gpi.GripperInterface):
    """
    This class is the second version of the Lite6 WRS grippers. It is designed to be used with the xArm Lite 6 robot.
    Hao developed the original code for this grippers.
    Weiwei kept it updated.
    author: hao, weiwei
    date: 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name="a2w_gripper"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = np.array([0.0, .05])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=2, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "finger_base.stl"),
            name=name + "_base",
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[14]
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, 0.17784])
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "finger.stl"),
                                                         name=name + "_finger1",
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.005)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, 1, 1])
        # the 2nd joint (right finger, -y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, 0])
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "finger.stl"),
                                                         name=name + "_finger2",
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.005)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([1, .5, .5, 1])
        # reinitialize
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .16])
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
        self.coupling.fix_to(pos=self._pos, rotmat=self._rotmat)
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def get_jaw_width(self):
        return self.jlc.jnts[1].motion_value

    @gpi.ei.EEInterface.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if self.jaw_range[0] / 2 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.goto_given_conf(jnt_values=[side_jawwidth, jaw_width])
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

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0], auto_rotate=False)
    mgm.gen_frame().attach_to(base)
    gripper = A2WGripper()
    gripper.change_jaw_width(0.05)
    gripper.gen_meshmodel(toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    # grippers.show_cdprimit()
    base.run()
