import os
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.robot_sim.manipulators.cobotta.ikgeo as ikgeo


class CVRB1213(mi.ManipulatorInterface):

    def __init__(self,
                 pos=rm.zeros(3),
                 rotmat=rm.np.eye(3),
                 home_conf=rm.zeros(6),
                 name="cvrb1213",
                 enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "cvrb1213_base.stl"), name=self.name + "_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # first joint and link
        self.jlc.jnts[0].loc_pos = rm.vec(0, 0, 0.21)
        self.jlc.jnts[0].loc_motion_ax = rm.vec(0, 0, 1)
        self.jlc.jnts[0].motion_range = rm.vec(-4.71238898038469, 4.71238898038469)
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j1.stl"),
                                                         name=self.name + "_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # second joint and link
        self.jlc.jnts[1].loc_pos = rm.vec(0.2, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = rm.vec(1, 0, 0)
        self.jlc.jnts[1].motion_range = rm.vec(-2.6179938779914944, 2.6179938779914944)
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j2.stl"),
                                                         name=self.name + "_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # third joint and link
        self.jlc.jnts[2].loc_pos = rm.vec(0, 0, 0.71)
        self.jlc.jnts[2].loc_motion_ax = rm.vec(1, 0, 0)
        self.jlc.jnts[2].motion_range = rm.vec(-2.6179938779914944, 2.6179938779914944)
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j3.stl"),
                                                         name=self.name + "_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = rm.vec(-0.25, -0.59, 0)
        self.jlc.jnts[3].loc_motion_ax = rm.vec(0, 1, 0)
        self.jlc.jnts[3].motion_range = rm.vec(-4.71238898038469, 4.71238898038469)
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j4.stl"),
                                                         name=self.name + "_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = rm.vec(0.15, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = rm.vec(1, 0, 0)
        self.jlc.jnts[4].motion_range = rm.vec(-2.6179938779914944, 2.6179938779914944)
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j5.stl"),
                                                         name=self.name + "_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = rm.vec(0, -0.16, 0)
        self.jlc.jnts[5].loc_motion_ax = rm.vec(0, 1, 0)
        self.jlc.jnts[5].motion_range = rm.vec(-6.283185307179586, 6.283185307179586)
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "cvrb1213_j6.stl"),
                                                         name=self.name + "_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = rm.vec(.7, .7, .7, 1.0)
        # flange
        self.jlc.set_flange(loc_flange_pos=rm.vec(0, 0, 0),
                            loc_flange_rotmat=rm.rotmat_from_euler(rm.pi / 2, 0, 0))
        # tcp
        self.loc_tcp_pos = rm.vec(0, 0, 0)
        self.loc_tcp_rotmat = rm.np.eye(3)
        self.jlc.finalize(ik_solver='dr', identifier_str=name+"r")
        # set up cc
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
        from_list = [l4, l5]
        into_list = [lb, l0, l1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        toggle_update = False
        # directly use specified ik
        self.jlc._ik_solver._k_max = 5
        rel_rotmat = tgt_rotmat @ self.loc_tcp_rotmat.T
        rel_pos = tgt_pos - tgt_rotmat @ self.loc_tcp_pos
        result = self.jlc.ik(tgt_pos=rel_pos, tgt_rotmat=rel_rotmat, seed_jnt_values=seed_jnt_values)
        if result is None:
            # mcm.mgm.gen_myc_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
            result = ikgeo.ik(jlc=self.jlc, tgt_pos=rel_pos, tgt_rotmat=rel_rotmat, seed_jnt_values=None)
            if result is None:
                print("No valid solutions found")
                return None
            else:
                if toggle_update:
                    rel_pos, rel_rotmat = rm.rel_pose(self.jlc.pos, self.jlc.rotmat, rel_pos, rel_rotmat)
                    rel_rotvec = self.jlc._ik_solver._rotmat_to_vec(rel_rotmat)
                    query_point = rm.np.concatenate((rel_pos, rel_rotvec))
                    # update dd driven file
                    tree_data = rm.np.vstack((self.jlc._ik_solver.query_tree.data, query_point))
                    self.jlc._ik_solver.jnt_data.append(result)
                    self.jlc._ik_solver.query_tree = scipy.spatial.cKDTree(tree_data)
                    print(f"Updating query tree, {id} explored...")
                    self.jlc._ik_solver.persist_data()
                return result
        else:
            return result


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 0, 0], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    arm = CVRB1213(enable_cc=True)
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    arm.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=False, toggle_flange_frame=False).attach_to(base)
    base.run()
