import os
import numpy as np
import modeling.collision_model as mcm
import basis.robot_math as rm
import robot_sim.manipulators.manipulator_interface as mi


class UR3E(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), home_conf=np.zeros(6), name='ur3e', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"))
        self.jlc.anchor.lnk_list[0].loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi)
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.5, .5, .5, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .15185])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "shoulder.stl"))
        self.jlc.jnts[0].lnk.loc_rotmat = rm.rotmat_from_euler(0, 0, np.pi)
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .0, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(1.570796327, 0, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[1].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "upperarm.stl"))
        self.jlc.jnts[1].lnk.loc_pos = np.array([.0, .0, 0.12])
        self.jlc.jnts[1].lnk.loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0, -np.pi / 2)
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([-0.24355, .0, .0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].motion_range = np.array([-np.pi, np.pi])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "forearm.stl"))
        self.jlc.jnts[2].lnk.loc_pos = np.array([.0, .0, 0.027])
        self.jlc.jnts[2].lnk.loc_rotmat = rm.rotmat_from_euler(np.pi / 2, 0, -np.pi / 2)
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.35, .35, .35, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.2132, .0, .13105])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist1.stl"))
        self.jlc.jnts[3].lnk.loc_pos = np.array([.0, .0, -0.104])
        self.jlc.jnts[3].lnk.loc_rotmat = rm.rotmat_from_euler(np.pi / 2, .0, .0)
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, -.08535, .0])
        self.jlc.jnts[4].loc_rotmat = rm.rotmat_from_euler(1.570796327, 0, 0)
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist2.stl"))
        self.jlc.jnts[4].lnk.loc_pos = np.array([.0, .0, -0.08535])
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, .0921, .0])
        self.jlc.jnts[5].loc_rotmat = rm.rotmat_from_euler(1.570796327, 3.141592653589793, 3.141592653589793)
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-np.pi * 2, np.pi * 2])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist3.stl"))
        self.jlc.jnts[5].lnk.loc_pos = np.array([.0, .0, -0.0921])
        self.jlc.jnts[5].lnk.loc_rotmat = rm.rotmat_from_euler(np.pi / 2, .0, .0)
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.5, .5, .5, 1.0])
        self.jlc.finalize(ik_solver='d', identifier_str=name)
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


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import time

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    arm = UR3E(enable_cc=True)
    arm_mesh = arm.gen_meshmodel()
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)

    tgt_pos = np.array([.25, .1, .1])
    tgt_rotmat = rm.rotmat_from_euler(0, np.pi, 0)
    mgm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    print(jnt_values)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        arm.goto_given_conf(jnt_values=jnt_values)
    arm_mesh = arm.gen_meshmodel(alpha=.3)
    arm_mesh.attach_to(base)
    tmp_arm_stick = arm.gen_stickmodel(toggle_flange_frame=True)
    tmp_arm_stick.attach_to(base)
    base.run()
