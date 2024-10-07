import os
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi

class FrankaResearch3Arm(mi.ManipulatorInterface):

    def __init__(self,
                 pos=rm.np.zeros(3),
                 rotmat=rm.np.eye(3),
                 ik_solver='d',
                 home_conf=rm.np.zeros(7),
                 name='franka_research_3_arm',
                 enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link0.dae"))
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = rm.np.array([0, 0, 0.333])
        self.jlc.jnts[0].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[0].motion_range = rm.np.array([-2.8973, 2.8973])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link1.dae"))
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = rm.np.array([0, 0, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(-1.57079632679, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[1].motion_range = rm.np.array([-1.8326, 1.8326])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link2.dae"))
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = rm.np.array([0, -0.316, 0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(1.57079632679, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[2].motion_range = rm.np.array([-2.8972, 2.8972])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link3.dae"))
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = rm.np.array([0.0825, 0, 0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.57079632679, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[3].motion_range = rm.np.array([-3.0718, -0.1222])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link4.dae"))
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = rm.np.array([-0.0825, 0.384, 0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-1.57079632679, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[4].motion_range = rm.np.array([-2.8798, 2.8798])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link5.dae"))
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = rm.np.array([0, 0, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.57079632679, 0, 0)
        self.jlc.jnts[5].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[5].motion_range = rm.np.array([0.4364, 4.6251])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link6.dae"))
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # seventh joint and link
        self.jlc.jnts[6].loc_pos = rm.np.array([0.088, 0, 0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(1.57079632679, 0, 0)
        self.jlc.jnts[6].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[6].motion_range = rm.np.array([-3.0543, 3.0543])
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "link7.dae"))
        self.jlc.jnts[6].lnk.cmodel.rgba = rm.np.array([.7, .7, .7, 1.0])
        # flange
        self.jlc.set_flange(loc_flange_pos=rm.np.array([0, 0, 0.107]), loc_flange_rotmat=rm.np.eye(3))
        # tcp
        self.loc_tcp_pos = rm.np.array([0, 0, 0])
        self.loc_tcp_rotmat = rm.np.eye(3)
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # setup cc
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l5, l6]
        into_list = [lb, l0, l1, l2]
        self.cc.set_cdpair_by_ids(from_list, into_list)

if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[3, 0, 4], lookat_pos=[0, 0, 0.75])
    mgm.gen_frame().attach_to(base)
    arm = FrankaResearch3Arm(enable_cc=True)
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    arm.gen_stickmodel(toggle_flange_frame=True).attach_to(base)
    # arm.show_cdprim()

    base.run()