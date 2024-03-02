import os
import numpy as np
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import modeling.model_collection as mmc
import robot_sim._kinematics.jlchain as rkjlc
import basis.robot_math as rm
import basis.constant as bc
import robot_sim.end_effectors.gripper.gripper_interface as gp
import robot_sim._kinematics.constant as rkc
import robot_sim._kinematics.model_generator as rkmg
import modeling.constant as mc


class CobottaGripper(gp.GripperInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 cdmesh_type=mc.CDMType.DEFAULT,
                 name="cobotta_gripper",
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        current_file_dir = os.path.dirname(__file__)
        self.coupling.finalize()
        # jlc
        self.jlc = rkjlc.JLChain(pos=self.coupling.gl_flange_pos, rotmat=self.coupling.gl_flange_rotmat, n_dof=2,
                                 name=name)
        # anchor
        self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "gripper_base.dae"),
                                                        cdmesh_type=self.cdmesh_type)
        self.jlc.anchor.lnk.cmodel.rgba = np.array([.35, .35, .35, 1])
        # the 1st joint (left finger)
        self.jlc.jnts[0].change_type(rkc.JntType.PRISMATIC, np.array([0, self.jaw_range[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[0].loc_motion_ax = bc.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "left_finger.dae"),
                                                         cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # the 2nd joint (right finger)
        self.jlc.jnts[1].change_type(rkc.JntType.PRISMATIC, np.array([0, -self.jaw_range[1]]))
        self.jlc.jnts[1].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[1].loc_motion_ax = bc.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "right_finger.dae"),
                                                         cdmesh_type=self.cdmesh_type)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # acting center
        self.loc_acting_center_pos = np.array([0, 0, 0.05])
        # reinitialize
        self.jlc.finalize()
        # jaw range
        self.jaw_range = np.array([0.0, .03])
        # collisions
        self.cdmesh_elements = [self.jlc.anchor.lnk,
                                self.jlc.jnts[0].lnk,
                                self.jlc.jnts[1].lnk]
        # self.enable_cc(toggle_cdprimit=enable_cc)

    # def enable_cc(self, toggle_cdprimit):
    #     if toggle_cdprimit:
    #         super().enable_cc()
    #         # cdprimit
    #         self.cc.add_cdlnks(self.jlc, [0, 1, 2])
    #         activelist = [self.jlc.lnks[0],
    #                       self.jlc.lnks[1],
    #                       self.jlc.lnks[2]]
    #         self.cc.set_active_cdlnks(activelist)
    #         self.all_cdelements = self.cc.cce_dict
    #     # cdmesh
    #     for cdelement in self.all_cdelements:
    #         cdmesh = cdelement['collision_model'].copy()
    #         self.cdmesh_collection.add_cm(cdmesh)

    def fix_to(self, pos, rotmat, jaw_width=None):
        self.pos = pos
        self.rotmat = rotmat
        if jaw_width is not None:
            side_jawwidth = jaw_width / 2.0
            if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
                self.jlc.jnts[0].motion_value = side_jawwidth
                self.jlc.jnts[1].motion_value = -jaw_width
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        self.jlc.fix_to(self.coupling.gl_flange_pos, self.coupling.gl_flange_rotmat)
        self.update_oiee()

    @gp.ei.assert_oiee_decorator
    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_range[1] / 2:
            self.jlc.go_given_conf(jnt_values=[side_jawwidth, -jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def get_jaw_width(self):
        return -self.jlc.jnts[1].motion_value

    def gen_stickmodel(self, toggle_tcp_frame=False, toggle_jnt_frames=False, name='cobotta_gripper_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_stick(self.coupling, toggle_jnt_frames=False, toggle_flange_frame=False).attach_to(m_col)
        rkmg.gen_jlc_stick(self.jlc, toggle_jnt_frames=toggle_jnt_frames, toggle_flange_frame=False).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='cobotta_gripper_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_mesh(self.coupling,
                          rgb=rgb,
                          alpha=alpha,
                          toggle_flange_frame=False,
                          toggle_jnt_frames=False,
                          toggle_cdmesh=toggle_cdmesh,
                          toggle_cdprim=toggle_cdprim).attach_to(m_col)
        rkmg.gen_jlc_mesh(self.jlc,
                          rgb=rgb,
                          alpha=alpha,
                          toggle_flange_frame=False,
                          toggle_jnt_frames=toggle_jnt_frames,
                          toggle_cdmesh=toggle_cdmesh,
                          toggle_cdprim=toggle_cdprim).attach_to(m_col)
        if toggle_tcp_frame:
            self._toggle_tcp_frame(m_col)
        # oiee
        self.gen_oiee_meshmodel(m_col, rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                toggle_cdmesh=toggle_cdmesh, toggle_frame=toggle_jnt_frames)
        return m_col


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[.5, .5, .5], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = CobottaGripper(cdmesh_type=mc.CDMType.OBB, enable_cc=True)
    grpr.fix_to(pos=np.array([0, .1, .1]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .7))
    print(grpr.grip_at_by_twovecs(jaw_center_pos=np.array([0, .1, .1]), approaching_vec=np.array([0, -1, 0]),
                                  fgr0_opening_vec=np.array([1, 0, 0]), jaw_width=.01))
    # grpr.change_jaw_width(.013)
    grpr.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=False, toggle_cdprim=False).attach_to(base)
    # # grpr.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    # grpr.gen_meshmodel().attach_to(base)
    # # grpr.gen_stickmodel().attach_to(base)
    # grpr.show_cdmesh()
    # grpr.show_cdprimit()
    base.run()
