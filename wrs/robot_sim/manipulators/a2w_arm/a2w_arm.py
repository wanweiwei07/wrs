import os
import wrs.basis.robot_math as rm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.modeling.collision_model as mcm


class A2WLeftArm(mi.ManipulatorInterface):
    """
    author: ziqi.xu, revised by weiwei
    date: 20221112, 20241031
    """

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3), ik_solver=None, name='a2warm', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=rm.np.zeros(7), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        # self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
        #     initor=os.path.join(current_file_dir, "meshes", "base_link.stl"), name="diana7_base")
        # self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.silver_gray
        # first joint and link
        self.jlc.jnts[0].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[0].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l1.stl"), name=name+"_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.silver_gray
        # second joint and link
        self.jlc.jnts[1].loc_pos = rm.np.array([0, 0, .21])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[1].motion_range = rm.vec(-1.8325, 1.8325)
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l2.stl"), name=name+"_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.silver_gray
        # third joint and link
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[2].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l3.stl"), name=name+"_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.dim_gray
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = rm.np.array([0, 0, 0.3285])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[3].motion_range = rm.vec(-2.5307, 0.5235)
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l4.stl"), name=name+"_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.silver_gray
        # fifth joint and link
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[4].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l5.stl"), name=name+"_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.dim_gray
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = rm.np.array([0,0,0.235])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, 0)
        self.jlc.jnts[5].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[5].motion_range = rm.vec(-1.8325, 1.8325)
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l6.stl"), name=name+"_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.silver_gray
        # seventh joint and link
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[6].loc_motion_ax = rm.const.x_ax
        self.jlc.jnts[6].motion_range = rm.vec(0,0)
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "l7.stl"), name=name+"_link7")
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
        # lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l5, l6]
        into_list = [l0, l1, l2]
        self.cc.set_cdpair_by_ids(from_list, into_list)


class A2WRightArm(mi.ManipulatorInterface):
    """
    author: ziqi.xu, revised by weiwei
    date: 20221112, 20241031
    """

    def __init__(self, pos=rm.np.zeros(3), rotmat=rm.np.eye(3), ik_solver=None, name='a2warm', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=rm.np.zeros(7), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        # self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
        #     initor=os.path.join(current_file_dir, "meshes", "base_link.stl"), name="diana7_base")
        # self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.const.silver_gray
        # first joint and link
        self.jlc.jnts[0].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[0].loc_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
        self.jlc.jnts[0].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r1.stl"), name=name+"_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.const.silver_gray
        # second joint and link
        self.jlc.jnts[1].loc_pos = rm.np.array([0, 0, .21])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[1].motion_range = rm.vec(-1.8325, 1.8325)
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r2.stl"), name=name+"_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.const.silver_gray
        # third joint and link
        self.jlc.jnts[2].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[2].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[2].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r3.stl"), name=name+"_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.silver_gray
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = rm.np.array([0, 0, 0.3285])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, 0)
        self.jlc.jnts[3].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[3].motion_range = rm.vec(-2.5307, 0.5235)
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r4.stl"), name=name+"_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.const.silver_gray
        # fifth joint and link
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[4].motion_range = rm.vec(-2*rm.pi, 2*rm.pi)
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r5.stl"), name=name+"_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.const.silver_gray
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = rm.np.array([0,0,0.235])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(-rm.pi/2, 0, rm.pi)
        self.jlc.jnts[5].loc_motion_ax = rm.const.z_ax
        self.jlc.jnts[5].motion_range = rm.vec(-1.8325, 1.8325)
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r6.stl"), name=name+"_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.const.silver_gray
        # seventh joint and link
        self.jlc.jnts[6].loc_rotmat = rm.rotmat_from_euler(rm.pi/2, 0, 0)
        self.jlc.jnts[6].loc_motion_ax = rm.const.x_ax
        self.jlc.jnts[6].motion_range = rm.vec(0,0)
        self.jlc.jnts[6].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "r7.stl"), name=name+"_link7")
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
        # lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        l6 = self.cc.add_cce(self.jlc.jnts[6].lnk)
        from_list = [l5, l6]
        into_list = [l0, l1, l2]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[3, 0, 2], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    manipulator = A2WLeftArm(enable_cc=True)
    manipulator.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    manipulator.gen_meshmodel(toggle_flange_frame=True, toggle_cdprim=False, alpha=.3).attach_to(base)
    # tgt_pos = rm.np.array([0.5, 0.3, 0.3])
    # tgt_rotmat = rm.rotmat_from_axangle(ax=rm.const.y_ax, angle=rm.pi)
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # jnt_angles = manipulator.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    # if jnt_angles is not None:
    #     manipulator.goto_given_conf(jnt_values=jnt_angles)
    #     manipulator.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True, alpha=.7).attach_to(base)
    base.run()
