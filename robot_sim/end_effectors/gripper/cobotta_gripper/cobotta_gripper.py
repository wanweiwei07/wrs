import os
import numpy as np
import modeling.collision_model as mcm
import modeling.geometric_model as mgm
import modeling.model_collection as mmc
import robot_sim._kinematics.jlchain as jl
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
                 name="cobott_gripper",
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, cdmesh_type=cdmesh_type, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos, cpl_end_rotmat = self.coupling.get_gl_tcp()
        # jaw range
        self.jaw_rng = np.array([0.0, .03])
        # jlc
        self.jlc = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, n_dof=2, name=name)
        # anchor
        self.jlc.anchor.pos = cpl_end_pos
        self.jlc.anchor.rotmat = cpl_end_rotmat
        self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "gripper_base.dae"))
        self.jlc.anchor.lnk.cmodel.rgba = np.array([.35, .35, .35, 1])
        # the 1st joint (left finger)
        self.jlc.jnts[0].change_type(rkc.JntType.PRISMATIC, np.array([0, self.jaw_rng[1] / 2]))
        self.jlc.jnts[0].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[0].loc_motion_ax = bc.y_ax
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "left_finger.dae"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # the 2nd joint (right finger)
        self.jlc.jnts[1].change_type(rkc.JntType.PRISMATIC, np.array([0, -self.jaw_rng[1]]))
        self.jlc.jnts[1].loc_pos = np.array([0, .0, .0])
        self.jlc.jnts[1].loc_motion_ax = bc.y_ax
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "right_finger.dae"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.5, .5, .5, 1])
        # action center
        self.action_center_pos = np.array([0, 0, .05])
        # reinitialize
        self.jlc.finalize(ik_solver=None)
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
            if 0 <= side_jawwidth <= self.jaw_rng[1] / 2:
                self.jlc.jnts[0].motion_val = side_jawwidth
                self.jlc.jnts[1].motion_val = -jaw_width
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos, cpl_end_rotmat = self.coupling.get_gl_tcp()
        self.jlc.fix_to(cpl_end_pos, cpl_end_rotmat)

    def change_jaw_width(self, jaw_width):
        side_jawwidth = jaw_width / 2.0
        if 0 <= side_jawwidth <= self.jaw_rng[1] / 2:
            self.jlc.go_given_conf(jnt_vals=[side_jawwidth, -jaw_width])
        else:
            raise ValueError("The angle parameter is out of range!")

    def get_jaw_width(self):
        return -self.jlc.jnts[1].motion_val

    def gen_stickmodel(self,
                       tgl_tcp_frame=False,
                       tgl_jnt_frame=False,
                       name='stick_model'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_stick(self.coupling, tgl_tcp_frame=False, tgl_jnt_frame=False).attach_to(m_col)
        rkmg.gen_jlc_stick(self.jlc, tgl_tcp_frame=False, tgl_jnt_frame=tgl_jnt_frame).attach_to(m_col)
        if tgl_tcp_frame:
            action_center_gl_pos = self.rotmat.dot(self.action_center_pos) + self.pos
            action_center_gl_rotmat = self.rotmat.dot(self.action_center_rotmat)
            rkmg.gen_tcp_frame(spos=self.pos,
                               tcp_gl_pos=action_center_gl_pos,
                               tcp_gl_rotmat=action_center_gl_rotmat).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      tgl_tcp_frame=False,
                      tgl_jnt_frame=False,
                      rgba=None,
                      tgl_cdprimitive=False,
                      tgl_cdmesh=False,
                      name='mesh_model'):
        m_col = mmc.ModelCollection(name=name)
        rkmg.gen_jlc_mesh(self.coupling,
                          tgl_tcp_frame=False,
                          tgl_jnt_frame=False,
                          tgl_cdmesh=tgl_cdmesh,
                          tgl_cdprimitive=tgl_cdprimitive).attach_to(m_col)
        rkmg.gen_jlc_mesh(self.jlc,
                          tgl_tcp_frame=False,
                          tgl_jnt_frame=tgl_jnt_frame,
                          tgl_cdmesh=tgl_cdmesh,
                          tgl_cdprimitive=tgl_cdprimitive).attach_to(m_col)
        if tgl_tcp_frame:
            action_center_gl_pos = self.rotmat.dot(self.action_center_pos) + self.pos
            action_center_gl_rotmat = self.rotmat.dot(self.action_center_rotmat)
            rkmg.gen_tcp_frame(spos=self.pos,
                               tcp_gl_pos=action_center_gl_pos,
                               tcp_gl_rotmat=action_center_gl_rotmat).attach_to(m_col)
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
    grpr = CobottaGripper(enable_cc=True)
    grpr.change_jaw_width(.013)
    grpr.gen_meshmodel(tgl_tcp_frame=True, tgl_jnt_frame=True, tgl_cdprimitive=True).attach_to(base)
    # # grpr.gen_stickmodel(tgl_jnt_frame=True).attach_to(base)
    # grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .7))
    # grpr.gen_meshmodel().attach_to(base)
    # # grpr.gen_stickmodel().attach_to(base)
    # # grpr.show_cdmesh()
    # # grpr.show_cdprimit()
    base.run()
