import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.ur3.ur3 as ur3
import wrs.robot_sim.end_effectors.grippers.robotiq85.robotiq85 as rtq85


class UR3_Rtq85(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), ik_solver='a', name="ur3_rtq85", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = ur3.UR3(pos=self.pos, rotmat=self.rotmat,
                                   ik_solver=ik_solver, name=name + "_manipulator", enable_cc=False)
        self.end_effector = rtq85.Robotiq85(pos=self.manipulator.gl_flange_pos,
                                            rotmat=self.manipulator.gl_flange_rotmat,
                                            coupling_offset_pos=np.array([.0, .0, .0484]),
                                            coupling_offset_rotmat=rm.rotmat_from_euler(.0, .0, np.pi/2),
                                            name=name + "_eef")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # TODO when pose is changed, oih info goes wrong
        # ee
        elb = self.cc.add_cce(self.end_effector.palm.lnk_list[0])
        ell0 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[0].lnk)
        ell1 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[1].lnk)
        ell2 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[2].lnk)
        ell3 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[3].lnk)
        elr0 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[0].lnk)
        elr1 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[1].lnk)
        elr2 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[2].lnk)
        elr3 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[3].lnk)
        # manipulator
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk, toggle_extcd=False)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, ell0, ell1, ell2, ell3, elr0, elr1, elr2, elr3, ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        oiee_into_list = []
        # TODO oiee?

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = UR3_Rtq85(enable_cc=True)
    robot.change_jaw_width(.05)
    # robot.cc.show_cdprim()
    robot.goto_given_conf(jnt_values=np.radians(np.array([20, -90, 120, 30, 0, 40, 0])))
    robot.cc.show_cdprim()
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], np.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print("IK result: ", jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
        robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    robot.show_cdprim()
    robot.unshow_cdprim()
    base.run()
