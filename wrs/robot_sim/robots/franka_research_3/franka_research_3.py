import numpy as np
import wrs.robot_sim.manipulators.franka_research_3_arm.franka_research_3_arm as fr3a
import wrs.robot_sim.end_effectors.grippers.franka_hand.franka_hand as frg
import wrs.robot_sim.robots.single_arm_robot_interface as rsi


class FrankaResearch3(rsi.SglArmRobotInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name='franka_research_3',
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(7)
        home_conf[1] = -0.785398163
        home_conf[3] = -2.35619449
        home_conf[5] = 1.57079632679
        home_conf[6] = 0.785398163397
        self.manipulator = fr3a.FrankaResearch3Arm(pos=self.pos, rotmat=self.rotmat, home_conf=home_conf,
                                                   name=name + "_manipulator", enable_cc=enable_cc)
        self.end_effector = frg.FrankaHand(pos=self.manipulator.gl_flange_pos,
                                           rotmat=self.manipulator.gl_flange_rotmat,
                                           name=name + "_eef")
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    @property
    def pos(self):
        return self._pos

    @pos.setter
    def pos(self, pos):
        self._pos = pos

    @property
    def rotmat(self):
        return self._rotmat

    @rotmat.setter
    def rotmat(self, rotmat):
        self._rotmat = rotmat

    def setup_cc(self):
        # end effector
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        ml6 = self.cc.add_cce(self.manipulator.jlc.jnts[6].lnk)
        from_list = [elb, el0, el1, ml5, ml6]
        into_list = [mlb, ml0, ml1, ml2]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 2, 0.5], lookat_pos=[0, 0, 0.5])
    mgm.gen_frame().attach_to(base)
    robot = FrankaResearch3(enable_cc=True)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    base.run()