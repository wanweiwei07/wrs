import os
import numpy as np
from panda3d.core import CollisionNode, CollisionBox, Point3, NodePath
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.grippers.gripper_interface as gpi
import time


class WRSGripper6(gpi.GripperInterface):
    """
    This class is the sixth version of Junbo's extension grippers. It is designed to be used with Nova2.
    author: weiwei
    date: 20240909
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name="wrs_gripper6"):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jaw range
        self.jaw_range = np.array([0.0, .246])
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=6, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r11.stl"),
                                                                name="wg_v6_r11",
                                                                cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.dim_gray
        # rack length
        self._rack_length = 0.041
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 3]))
        self.jlc.jnts[0].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r21.stl"),
                                                         name="wg_v6_r21",
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._mid_cdprim, ex_radius=.002)
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.tab20_list[5]
        # the 2nd joint (left finger, +y direction)
        self.jlc.jnts[1].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[1].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r12.stl"),
                                                         name="wg_v6_r12",
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.002)
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.tab20_list[1]
        # the 3nd joint (left finger, +y direction)
        self.jlc.jnts[2].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[2].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r22.stl"),
                                                         name="wg_v6_r22",
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._fgr_cdprim, ex_radius=.002)
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.tab20_list[3]
        # the 1st joint (rgt finger, -y direction)
        self.jlc.jnts[3].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 3]))
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(.0, .0, np.pi)
        self.jlc.jnts[3].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r21.stl"),
                                                         name="wg_v6_r21",
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._mid_cdprim, ex_radius=.002)
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.tab20_list[5]
        # the 2nd joint (rgt finger, -y direction)
        self.jlc.jnts[4].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[4].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r12.stl"),
                                                         name="wg_v6_r12",
                                                         cdmesh_type=self.cdmesh_type, ex_radius=.002)
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.tab20_list[1]
        # the 3nd joint (rgt finger, -y direction)
        self.jlc.jnts[5].change_type(rkjlc.const.JntType.PRISMATIC, motion_range=np.array([0.0, self.jaw_range[1]]))
        self.jlc.jnts[5].loc_motion_ax = rm.const.y_ax
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "v6_r22.stl"),
                                                         name="wg_v6_r22",
                                                         cdmesh_type=self.cdmesh_type,
                                                         cdprim_type=mcm.const.CDPrimType.USER_DEFINED,
                                                         userdef_cdprim_fn=self._fgr_cdprim, ex_radius=.002)
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.tab20_list[3]
        self.jlc.finalize()
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, .179])
        # collision detection
        # collisions
        self.cdelements = (self.jlc.anchor.lnk_list[0],
                           self.jlc.jnts[0].lnk,
                           self.jlc.jnts[2].lnk,
                           self.jlc.jnts[3].lnk,
                           self.jlc.jnts[5].lnk)

    @staticmethod
    def _fgr_cdprim(name="wg_v6_fgr", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(.0, 0.003, .129),
                                              x=.01 + ex_radius, y=0.003 + ex_radius, z=.07 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        collision_primitive_c1 = CollisionBox(Point3(-.0265, -0.0185, .0745),
                                              x=.02 + ex_radius, y=0.025 + ex_radius, z=.0165 + ex_radius)
        pdcnd.addSolid(collision_primitive_c1)
        cdprim = NodePath(name + "_cdprim")
        cdprim.attachNewNode(pdcnd)
        return cdprim

    @staticmethod
    def _mid_cdprim(name="wg_v6_mid", ex_radius=None):
        pdcnd = CollisionNode(name + "_cnode")
        collision_primitive_c0 = CollisionBox(Point3(-.0305, -.0165, .0445),
                                              x=.0145 + ex_radius, y=0.0285 + ex_radius, z=.04 + ex_radius)
        pdcnd.addSolid(collision_primitive_c0)
        cdprim = NodePath(name + "_cprim")
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
            # motion_rack_length = self._rack_length - side_jawwidth / 3.0
            motion_rack_length = side_jawwidth / 3.0
            self.jlc.goto_given_conf(jnt_values=[motion_rack_length, motion_rack_length, motion_rack_length,
                                                 4 * motion_rack_length, motion_rack_length, motion_rack_length])
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
    import wrs.robot_sim._kinematics.model_generator as mg

    base = wd.World(cam_pos=[-.5, .0, .08], lookat_pos=[0, 0, 0.08], auto_rotate=False)
    mcm.mgm.gen_frame().attach_to(base)
    gripper = WRSGripper6()
    gripper_node_list = []
    gripper.change_jaw_width(0)
    # gripper.gen_meshmodel(alpha=0.3,toggle_tcp_frame=True, toggle_cdprim=True).attach_to(base)
    # mg.gen_lnk_mesh(gripper.jlc.jnts[0].lnk,toggle_cdprim=True).attach_to(base)
    # mg.gen_lnk_mesh(gripper.jlc.jnts[2].lnk,toggle_cdprim=True).attach_to(base)


    for i in range(30):
        gripper.change_jaw_width(.008*i)
        gripper_node_list.append(gripper.gen_meshmodel(toggle_tcp_frame=False, toggle_cdprim=True))
    class AnimeData(object):
        def __init__(self, nodes_rbt):
            self.counter = 0
            self.nodes_rbt = nodes_rbt
            self.model = nodes_rbt[self.counter]
    def update(anime_data, task):
        # if base.inputmgr.keymap["space"] is True:
            time.sleep(.1)
            if anime_data.counter >= len(anime_data.nodes_rbt):
                anime_data.model.detach()
                # anime_data.support_facets[anime_data.counter - 1].detach()
                anime_data.counter = 0
            else:
                anime_data.model.detach()
                anime_data.model = anime_data.nodes_rbt[anime_data.counter]
                # anime_data.model.alpha = .3
                anime_data.model.attach_to(base)
                anime_data.counter += 1
            return task.cont

    anime_data = AnimeData(gripper_node_list)
    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()
