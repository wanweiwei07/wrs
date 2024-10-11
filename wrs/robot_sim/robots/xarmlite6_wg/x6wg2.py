import numpy as np
import wrs.robot_sim.manipulators.xarm_lite6 as manipulator
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v2 as end_effector
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.modeling.geometric_model as mgm


class XArmLite6WG2(sari.SglArmRobotInterface):
    """
    Simulation for the XArm Lite 6 With the WRS grippers
    Author: Chen Hao (chen960216@gmail.com), Updated by Weiwei
    Date: 20220925osaka, 20240318
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6_wrsgripper2', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = manipulator.XArmLite6(pos=pos, rotmat=rotmat, name="xarmlite6g2_arm",
                                                 enable_cc=False)
        self.end_effector = end_effector.WRSGripper2(pos=self.manipulator.gl_flange_pos,
                                                     rotmat=self.manipulator.gl_flange_rotmat, name="xarmlite6g2_hnd")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # ee
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
        from_list = [mlb, ml0, ml1]
        into_list = [elb, el0, el1, ml3, ml4, ml5]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = [el0, el1]

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
    from wrs import wd

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    mgm.gen_frame().attach_to(base)
    robot = XArmLite6WG2(pos=np.array([0,0,0]), enable_cc=True)
    robot.gen_meshmodel(alpha=.3).attach_to(base)
    robot.gen_stickmodel(toggle_jnt_frames=True, toggle_tcp_frame=True).attach_to(base)
    # tic = time.time()
    # trm_model=xarm_model.acquire_cm_trm()
    # toc = time.time()
    # print(toc-tic)
    # print(trm_model.vertices)
    # print("Is self collided?", robot.is_collided())

    goal_pos = np.array([0.30013906, -0.04995994, 0.04078924])
    goal_rotmat = np.array([[-0.36436617, -0.23050759, -0.90227687],
                            [-0.1049671, 0.97287324, -0.20615424],
                            [0.92532114, 0.01959376, -0.37867781]])
    mgm.gen_frame(pos=goal_pos, rotmat=goal_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=goal_pos, tgt_rotmat=goal_rotmat)
    if jnt_values is None:
        base.run()
    robot.goto_given_conf(jnt_values=jnt_values)
    robot.gen_meshmodel().attach_to(base)
    base.run()

    tgt_pos = np.array([0.2995316, -0.04995615, 0.1882039])
    tgt_rotmat = np.array([[0.03785788, 0.05806798, 0.99759455],
                           [0.01741114, 0.99812033, -0.05875933],
                           [-0.99913144, 0.01959376, 0.03677569]])
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos, tgt_rotmat)
    print(jnt_values)
    robot.goto_given_conf(jnt_values)
    robot.gen_meshmodel().attach_to(base)
    base.run()
