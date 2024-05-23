import os
import numpy as np
import robot_sim._kinematics.jlchain as jl
import basis.robot_math as rm
import robot_sim.end_effectors.single_contact.single_contact_interface as si
import modeling.collision_model as mcm
import modeling.model_collection as mmc


class ORSD(si.SCTInterface):
    """
     # orsd = OnRobot ScrewDriver
    author: weiwei
    date: 20230803
    """

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 coupling_offset_pos=np.zeros(3),
                 coupling_offset_rotmat=np.eye(3),
                 cdmesh_type=mcm.mc.CDMType.DEFAULT,
                 name='onrobot_screwdriver',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        self.coupling.loc_flange_pose_list[0] = (coupling_offset_pos, coupling_offset_rotmat)
        self.coupling.lnk_list[0].cmodel = mcm.gen_stick(spos=pos,
                                                         epos=self.coupling.gl_flange_pose_list[0][0],
                                                         type="rect",
                                                         radius=0.035,
                                                         rgb=np.array([.35, .35, .35]),
                                                         alpha=1,
                                                         n_sec=24)
        # jlc
        self.jlc = jl.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                              rotmat=self.coupling.gl_flange_pose_list[0][1],
                              n_dof=1, name='orsd_jlc')
        self.jlc.jnts[0].loc_pos = np.array([0.16855000, 0, 0.09509044])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "or_screwdriver.stl"),
            cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.55, .55, .55, 1])
        # reinitialize
        self.jlc.finalize()
        #  action center
        self.loc_acting_center_pos = self.coupling.gl_flange_pose_list[0][1] @ np.array(
            [0.16855000, 0, 0.09509044]) + coupling_offset_pos
        self.action_center_rotmat = self.coupling.gl_flange_pose_list[0][1]
        # collision detection
        self.cdmesh_elements = (self.jlc.anchor.lnk_list[0],
                                self.jlc.jnts[0].lnk)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            self.change_jaw_width(jaw_width=jaw_width)
        self.coupling.pos = self.pos
        self.coupling.rotmat = self.rotmat
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

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
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = ORSD(coupling_offset_pos=np.array([0, 0, 0.0145]), enable_cc=True)
    grpr.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    grpr.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    base.run()
