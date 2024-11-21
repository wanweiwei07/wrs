import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.cobotta.cvr038 as cbta
import wrs.robot_sim.end_effectors.grippers.cobotta_gripper.cobotta_gripper as cbtg


class Cobotta(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 3
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = cbta.CVR038(pos=self.pos, rotmat=self.rotmat, name=name + "_arm", enable_cc=False)
        self.manipulator.home_conf = home_conf
        self.end_effector = cbtg.CobottaGripper(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat, name=name + "_hnd")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # end effector
        ee_cces = []
        for id, cdlnk in enumerate(self.end_effector.cdelements):
            ee_cces.append(self.cc.add_cce(cdlnk))
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = ee_cces + [ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces[1:]

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()


if __name__ == '__main__':
    import time
    from wrs import rm, mcm, wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = Cobotta(pos=rm.vec(0.168, .3, 0), rotmat=rm.rotmat_from_euler(0, 0, rm.pi / 2), enable_cc=True)
    # robot.jaw_to(.02)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    tgt_pos = np.array([-0.12, .23, .058])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # tgt_rotmat = rm.rotmat_from_quaternion([0.707, -0.707, 0, 0])
    tgt_rotmat = rm.rotmat_from_quaternion([0.156, 0.988, 0, 0])
    mcm.mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot.show_cdprim()
    # robot.unshow_cdprim()
    base.run()

    robot.goto_given_conf(jnt_values=np.array([0, np.pi / 2, np.pi * 11 / 20, 0, np.pi / 2, 0]))
    robot.show_cdprim()

    box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos, rgb=np.array([1, 1, 0]), alpha=.3)
    box.attach_to(base)
    tic = time.time()
    result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
    print(result)
    toc = time.time()
    print(toc - tic)
    for pnt in contacts:
        mgm.gen_sphere(pnt).attach_to(base)

    base.run()
