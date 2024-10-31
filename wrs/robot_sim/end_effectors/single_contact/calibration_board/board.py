import os
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.robot_sim.end_effectors.single_contact.single_contact_interface as si


class Board(si.SCTInterface):

    def __init__(self,
                 pos=rm.np.zeros(3),
                 rotmat=rm.np.eye(3),
                 coupling_offset_pos=rm.np.zeros(3),
                 coupling_offset_rotmat=rm.np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                 name='sztu_sd'):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        # jlc (essentially an anchor since there is no joint)
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pose_list[0][0],
                                 rotmat=self.coupling.gl_flange_pose_list[0][1], n_dof=0, name='calib_board_jlc')
        self.jlc.anchor.lnk_list[0].name="calib_board"
        self.jlc.anchor.lnk_list[0].loc_rotmat=rm.rotmat_from_euler(rm.pi, 0,0)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "board.stl"), name="calib_board",
            cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.np.array([.9, .77, .52, 1.0])
        # reinitialize
        self.jlc.finalize()
        #  action center
        self.loc_acting_center_pos = self.coupling.loc_flange_pose_list[0][1] @ rm.np.array([0, -.2075, 0.009]) + self.coupling.loc_flange_pose_list[0][0]
        self.loc_acting_center_rotmat = self.coupling.loc_flange_pose_list[0][1]
        # collision detection
        self.cdelements = [self.jlc.anchor.lnk_list[0]]

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.fix_to(pos=self._pos, rotmat=self._rotmat)
        self.jlc.fix_to(self.coupling.gl_flange_pose_list[0][0], self.coupling.gl_flange_pose_list[0][1])
        self.update_oiee()

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False):
        m_col = mmc.ModelCollection(name=self.name+'_stickmodel')
        self.coupling.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=False).attach_to(m_col)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self, rgb=None, alpha=None, toggle_tcp_frame=False, toggle_jnt_frames=False,
                      toggle_cdprim=False, toggle_cdmesh=False):
        m_col = mmc.ModelCollection(name=self.name+'_meshmodel')
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
    from wrs import wd, mgm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    screwdriver = Board()
    screwdriver.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)

    # screwdriver.show_cdmesh()
    base.run()


