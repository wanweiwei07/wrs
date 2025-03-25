import os
import numpy as np
import scipy
import wrs.basis.robot_math as rm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi
import wrs.robot_sim.manipulators.cobotta.ikgeo as ikgeo


class CVR038(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cvr038", enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=np.zeros(6), name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_base.dae"), name=self.name + "_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.617994, 2.617994])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j1.dae"), name=self.name + "_link1")
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0, 0, 0.18])
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j2.dae"), name=self.name + "_link2")
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0, 0, 0.165])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([0.3141593, 2.443461])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j3.dae"), name=self.name + "_link3")
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.012, 0.02, 0.088])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-2.96706, 2.96706])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j4.dae"), name=self.name + "_link4")
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0, -.02, .0895])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[4].motion_range = np.array([-1.658063, 2.356194])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j5.dae"), name=self.name + "_link5")
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0, -.0445, 0.042])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-2.96706, 2.96706])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "cvr038_j6.dae"), name=self.name + "_link6")
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.finalize(ik_solver='s', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
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
        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def ik(self,
           tgt_pos,
           tgt_rotmat,
           seed_jnt_values=None,
           option="single",
           toggle_dbg=False):
        """
        This ik solver uses ikgeo to find an initial solution and then uses numik(pinv) as a backbone for precise
        computation. IKGeo assumes the jlc root is at pos=0 and rotmat=I. Numik uses jlc fk and does not have this
        assumption. IKGeo will shift jlc root to zero. There is no need to do them on the upper level. (20241121)
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param option:
        :param toggle_dbg:
        :return:
        """
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
                    query_point = np.concatenate((rel_pos, rel_rotvec))
                    # update dd driven file
                    tree_data = np.vstack((self.jlc._ik_solver.query_tree.data, query_point))
                    self.jlc._ik_solver.jnt_data.append(result)
                    self.jlc._ik_solver.query_tree = scipy.spatial.cKDTree(tree_data)
                    print(f"Updating query tree, {id} explored...")
                    self.jlc._ik_solver.persist_data()
                return result
        else:
            return result


if __name__ == '__main__':
    import time
    from wrs import wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    arm = CVR038()
    arm.gen_meshmodel(alpha=.3).attach_to(base)
    arm.gen_stickmodel().attach_to(base)
    # base.run()

    arm = CVR038(pos =rm.vec(0.168, .3, 0), rotmat = rm.rotmat_from_euler(0, 0, rm.pi / 2), enable_cc=True)
    arm.jlc.test_ik_success_rate()
    base.run()

    # arm.jlc._ik_solver.test_success_rate()
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    # base.run()

    # tgt_pos = np.array([.25, .1, .1])
    # tgt_rotmat = rm.rotmat_from_euler(0, np.pi, 0)
    tgt_pos = np.array([0.05, .53, .058])
    tgt_rotmat = rm.rotmat_from_quaternion([0.156, 0.988, 0, 0])
    mcm.mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        print(np.degrees(jnt_values))
        arm.goto_given_conf(jnt_values=jnt_values)
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    base.run()

    arm.goto_given_conf(jnt_values=np.array([0, np.pi / 2, np.pi * 3 / 4, 0, np.pi / 2, 0]))
    arm.show_cdprim()

    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)

    box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos)
    box.attach_to(base)
    tic = time.time()
    result, contacts = arm.is_collided(obstacle_list=[box], toggle_contacts=True)
    toc = time.time()
    print(toc - tic)
    for pnt in contacts:
        mcm.mgm.gen_sphere(pnt).attach_to(base)
    base.run()
