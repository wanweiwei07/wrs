import os
import math
import numpy as np
import modeling.collision_model as mcm
import basis.robot_math as rm
import robot_sim.manipulators.manipulator_interface as mi


class CobottaArm(mi.ManipulatorInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 home_conf=np.zeros(6),
                 name="cobotta_arm",
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "base_link.dae"))
        self.jlc.anchor.lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2.617994, 2.617994])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j1.dae"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([0, 0, 0.18])
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j2.dae"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([0, 0, 0.165])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j3.dae"))
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([-0.012, 0.02, 0.088])
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[3].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j4.dae"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([0, -.02, .0895])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[4].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j5.dae"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([0, -.0445, 0.042])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[5].motion_range = np.array([-1.047198, 1.745329])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "j6.dae"))
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        self.jlc.finalize(ik_solver='d', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, .05])
        self.loc_tcp_rotmat = rm.rotmat_from_axangle(np.array([0, 1, 0]), np.pi / 12)
        # collision detection
        # if enable_cc:
        #     self.enable_cc()

    # def enable_cc(self):
    #     super().enable_cc()
    #     self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6])
    #     activelist = [self.jlc.lnks[0],
    #                   self.jlc.lnks[1],
    #                   self.jlc.lnks[2],
    #                   self.jlc.lnks[3],
    #                   self.jlc.lnks[4],
    #                   self.jlc.lnks[5],
    #                   self.jlc.lnks[6]]
    #     self.cc.set_active_cdlnks(activelist)
    #     fromlist = [self.jlc.lnks[0],
    #                 self.jlc.lnks[1]]
    #     intolist = [self.jlc.lnks[3],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[2]]
    #     intolist = [self.jlc.lnks[4],
    #                 self.jlc.lnks[5],
    #                 self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.jlc.lnks[3]]
    #     intolist = [self.jlc.lnks[6]]
    #     self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, .3])
    gm.gen_frame().attach_to(base)
    tmp_arm = CobottaArm(enable_cc=False)
    # tmp_arm_mesh = tmp_arm.gen_meshmodel(alpha=.3)
    # tmp_arm_mesh.attach_to(base)
    # tmp_arm_stick = tmp_arm.gen_stickmodel(toggle_flange_frame=True)
    # tmp_arm_stick.attach_to(base)
    # base.run()
    tgt_pos = np.array([.25, .1, .1])
    tgt_rotmat = rm.rotmat_from_euler(0, np.pi, 0)
    gm.gen_dashed_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    tic = time.time()
    jnt_values = tmp_arm.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    toc = time.time()
    print(toc - tic)
    if jnt_values is not None:
        tmp_arm.goto_given_conf(jnt_values=jnt_values)
        tmp_arm_mesh = tmp_arm.gen_meshmodel(alpha=.3)
        tmp_arm_mesh.attach_to(base)
        tmp_arm_stick = tmp_arm.gen_stickmodel(toggle_flange_frame=True)
        tmp_arm_stick.attach_to(base)
    # tic = time.time()
    # print(manipulator_instance.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # mgm.GeometricModel("./meshes/base.dae").attach_to(base)
    # mgm.gen_frame().attach_to(base)
    base.run()
