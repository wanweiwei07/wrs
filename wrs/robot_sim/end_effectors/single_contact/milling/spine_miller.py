import os
import wrs.basis.robot_math as rm
import wrs.modeling.model_collection as mmc
import wrs.modeling.collision_model as mcm
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.robot_sim.end_effectors.single_contact.single_contact_interface as si


class SpineMiller(si.SCTInterface):

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3),
                 cdmesh_type=mcm.const.CDMeshType.DEFAULT, name='spine_miller', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.gl_flange_pose_list[0][0]
        cpl_end_rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.anchor = rkjl.Anchor(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_lnk=3)
        # flange 0 (FT sensor)
        self.anchor.lnk_list[0].cmodel = mcm.gen_stick(spos=rm.vec(0, 0, 0),
                                                       epos=rm.vec(0, 0, 0.05),
                                                       radius=.05)
        self.anchor.lnk_list[0].cmodel.rgba = rm.const.carrot_orange
        # flange 1 (motor housing)
        self.anchor.lnk_list[1].cmodel = mcm.gen_stick(spos=rm.vec(0, 0, 0),
                                                       epos=rm.vec(0, 0, 0.175),
                                                       type="rect",
                                                       radius=.02)
        self.anchor.lnk_list[1].cmodel.rgba = rm.const.gray
        # flange 2 (tool tip)
        self.anchor.lnk_list[2].cmodel = mcm.gen_stick(spos=rm.vec(0, 0, 0),
                                                       epos=rm.vec(0, 0, 0.22),
                                                       type="round",
                                                       radius=.005)
        self.anchor.lnk_list[2].cmodel.rgba = rm.const.antique_gold
        # action center
        self.loc_acting_center_pos = rm.np.array([0, 0, .22])
        # collision detection
        self.all_cdelements = [self.anchor.lnk_list[0],
                               self.anchor.lnk_list[1],
                               self.anchor.lnk_list[2]]

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.coupling.fix_to(self._pos, self._rotmat)
        cpl_end_pos = self.coupling.gl_flange_pose_list[0][0]
        cpl_end_rotmat = self.coupling.gl_flange_pose_list[0][1]
        self.anchor.fix_to(cpl_end_pos, cpl_end_rotmat)

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False,
                       toggle_flange_frame=False, name='_stickmodel'):
        m_col = mmc.ModelCollection(name=self.name + name)
        self.coupling.gen_stickmodel(toggle_flange_frame=toggle_flange_frame,
                                     toggle_root_frame=toggle_jnt_frames).attach_to(m_col)
        self.anchor.gen_stickmodel(toggle_flange_frame=toggle_tcp_frame).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='_meshmodel'):
        m_col = mmc.ModelCollection(name=self.name + name)
        self.coupling.gen_meshmodel(rgb=rgb,
                                    alpha=alpha,
                                    toggle_flange_frame=toggle_flange_frame,
                                    toggle_root_frame=False,
                                    toggle_cdmesh=toggle_cdmesh,
                                    toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.anchor.gen_meshmodel(rgb=rgb,
                                  alpha=alpha,
                                  toggle_flange_frame=toggle_flange_frame,
                                  toggle_root_frame=toggle_jnt_frames,
                                  toggle_cdmesh=toggle_cdmesh,
                                  toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    ee = SpineMiller()
    ee.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    base.run()
