import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm


class Diana7(mi.ManipulatorInterface):
    """
    author: ziqi.xu, revised by weiwei
    date: 20221112, 20241031
    """

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3), ik_solver='d', name='diana7', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=rm.np.zeros(7), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base_link.stl"), name="diana7_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.silver_gray
        # first joint and link
        self.jlc.jnts[0].loc_pos = rm.np.array([0, 0, 0.2856])
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(3.14159265358, 0.0000000, 0.00000)
        self.jlc.jnts[0].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[0].motion_range = rm.np.radians([-179, 179])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link1.stl"), name="diana7_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.silver_gray
        # second joint and link
        self.jlc.jnts[1].loc_pos = rm.np.array([0, 0, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.5707963267949, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[1].motion_range = rm.np.radians([-90, 90])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link2.stl"), name="diana7_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.silver_gray
        # third joint and link
        self.jlc.jnts[2].loc_pos = rm.np.array([0, -0.4586, 0])
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(-1.5707963267949, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[2].motion_range = rm.np.radians([-179, 179])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link3.stl"), name="diana7_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.dim_gray
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = rm.np.array([0.065, 0, 0])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(1.5707963267949, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[3].motion_range = rm.np.radians([0, 175])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link4.stl"), name="diana7_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.silver_gray
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = rm.np.array([-0.0528, -0.4554, 0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(-1.5707963267949, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[4].motion_range = rm.np.array([-179, 179])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link5.stl"), name="diana7_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.dim_gray
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = rm.np.array([-0.0122, 0, 0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.5707963267949, 0, 3.1416)
        self.jlc.jnts[5].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[5].motion_range = rm.np.array([-179, 179])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link6.stl"), name="diana7_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.silver_gray
        # seventh joint and link
        self.jlc.jnts[6].loc_pos = rm.np.array([0.087, -0.1169, 0])
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(-1.5707963267949, 0, 0)
        self.jlc.jnts[6].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[6].motion_range = rm.np.array([-179, 179])
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "link7.stl"), name="diana7_link7")
        self.jlc.jnts[6].lnk.cmodel.rgba = rm.const.silver_gray
        # flange
        self.jlc.set_flange(loc_flange_pos=rm.np.array([0, 0, 0]),
                            loc_flange_rotmat=rm.rotmat_from_axangle(rm.const.y_ax, rm.pi))
        # finalize
        self.jlc.finalize(ik_solver=ik_solver, identifier_str=name)
        # tcp
        self.loc_tcp_pos = rm.np.array([0, 0, 0])
        self.loc_tcp_rotmat = rm.np.eye(3)
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
    from wrs import wd, mgm

    base = wd.World(cam_pos=[3, 0, 2], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    manipulator = Diana7(enable_cc=True)
    manipulator.gen_meshmodel(toggle_flange_frame=True, toggle_cdprim=False).attach_to(base)
    tgt_pos = rm.np.array([0.5, 0.3, 0.3])
    tgt_rotmat = rm.rotmat_from_axangle(ax=rm.const.y_ax, angle=rm.pi)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_angles = manipulator.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_angles is not None:
        manipulator.goto_given_conf(jnt_values=jnt_angles)
        manipulator.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True, alpha=.7).attach_to(base)
    base.run()
