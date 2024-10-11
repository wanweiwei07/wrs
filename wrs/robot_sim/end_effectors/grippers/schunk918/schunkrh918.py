import os
import math
import numpy as np
import wrs.modeling.model_collection as mmc
from wrs import basis as rm, robot_sim as rkjlc, robot_sim as gpi, modeling as mcm, modeling as gm


class SchunkRH918(gpi.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMeshType.DEFAULT,
                 name='schunkrh918'):
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
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=4, name=name)
        # anchor
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi / 2)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"),
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.bc.tab20_list[14]
        # the 1st joint (left finger, +y direction)
        self.jlc.jnts[0].change_type(rkjlc.rkc.JntType.PRISMATIC, motion_range=np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([-.01, .04, .073])
        self.jlc.jnts[0].loc_motion_ax = rm.bc.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "slider.stl"),
            cdmesh_type=self.cdmesh_type, ex_radius=.005)
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, -np.pi / 2)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, 1, 1])
        # the 2nd joint (right finger, -y direction)
        self.jlc.jnts[1].loc_pos = np.array([.02, .008, 0])
        self.jlc.jnts[1].loc_motion_ax = rm.bc.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "finger.stl"),
            cdmesh_type=self.cdmesh_type, ex_radius=.005)
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(0, math.pi, math.pi/2)
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
        # # - lft
        # self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(2), name='base_lft_slider_finger')
        # self.lft.jnts[1]['loc_pos'] = np.array([-.01, .04, .073])
        # self.lft.jnts[1]['end_type'] = 'prismatic'
        # self.lft.jnts[1]['motion_range'] = [0, .025]
        # self.lft.jnts[1]['loc_motionax'] = np.array([0, -1, 0])
        # self.lft.lnks[0]['name'] = "base"
        # self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        # self.lft.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        # self.lft.lnks[0]['rgba'] = [.2, .2, .2, 1]
        # self.lft.lnks[1]['name'] = "slider1"
        # self.lft.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "slider.stl")
        # self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # self.lft.lnks[1]['loc_pos'] = np.zeros(3)
        # self.lft.lnks[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, -math.pi/2)
        # self.lft.jnts[2]['loc_pos'] = np.array([.02, .008, 0])
        # self.lft.lnks[2]['name'] = "finger1"
        # self.lft.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        # self.lft.lnks[2]['rgba'] = [.8, .8, .8, 1]
        # self.lft.lnks[2]['loc_pos'] = np.zeros(3)
        # self.lft.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(0, math.pi, math.pi/2)
        # # - rgt
        # self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, home_conf=np.zeros(2), name='rgt_finger')
        # self.rgt.jnts[1]['loc_pos'] = np.array([.01, -.04, .073])
        # self.rgt.jnts[1]['end_type'] = 'prismatic'
        # self.rgt.jnts[1]['loc_motionax'] = np.array([0, 1, 0])
        # self.rgt.lnks[1]['name'] = "slider2"
        # self.rgt.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "slider.stl")
        # self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # self.rgt.lnks[1]['loc_pos'] = np.zeros(3)
        # self.rgt.lnks[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi / 2)
        # self.rgt.jnts[2]['loc_pos'] = np.array([-.02, -.008, 0])
        # self.rgt.lnks[2]['name'] = "finger2"
        # self.rgt.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "finger.stl")
        # self.rgt.lnks[2]['rgba'] = [.8, .8, .8, 1]
        # self.rgt.lnks[2]['loc_pos'] = np.zeros(3)
        # self.rgt.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(0, math.pi, -math.pi / 2)
        # # jaw range
        # self.jaw_range = [0.0, .05]
        # # jaw center
        # self.jaw_center_pos = np.array([0, 0, .145])
        # # reinitialize
        # self.lft.finalize()
        # self.rgt.finalize()
        # # collision detection
        # self.all_cdelements=[]
        # self.enable_cc(toggle_cdprimit=enable_cc)


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
    gm.gen_frame().attach_to(base)
    grpr = SchunkRH918()
    # grpr.change_jaw_width(.03)
    # print("ee_values = ", grpr.get_jaw_width())
    grpr.gen_meshmodel().attach_to(base)
    base.run()
