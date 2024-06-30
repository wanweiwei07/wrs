import numpy as np
import robot_sim.manipulators.dobot_nova2.nova2 as nv2
import robot_sim.end_effectors.gripper.wrs_gripper.wrs_gripper_v3 as wg3
import robot_sim.robots.single_arm_robot_interface as rsi


class Nova2WG3(rsi.SglArmRobotInterface):
    """
    Simulation for the Nova2 with WRS gripper v3
    author: weiwei
    date: 20240630
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='nova2_wg3', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = nv2.Nova2(pos=pos, rotmat=rotmat, name="nv2wg3_arm", enable_cc=False)
        self.end_effector = wg3.WRSGripper3(pos=self.manipulator.gl_flange_pos,
                                            rotmat=self.manipulator.gl_flange_rotmat, name="nv2wg3_hand")
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
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, el0, el1, ml3, ml4, ml5]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3, ml4]
        return

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()


if __name__ == '__main__':
    import numpy as np
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import modeling.collision_model as mcm

    base = wd.World(cam_pos=[3, 0, 2], lookat_pos=[0, 0, .5])
    mgm.gen_frame().attach_to(base)
    robot = Nova2WG3(enable_cc=True)

    box = mcm.gen_box(xyz_lengths=[.1, .1, .1])
    box.pose = (np.array([.1, .1, .9]), np.eye(3))
    box.attach_to(base)

    robot.hold(obj_cmodel=box)
    robot.show_cdprim()
    robot_model = robot.gen_meshmodel()
    robot_model.attach_to(base)
    print("Is self collided?", robot.is_collided())

    robot.goto_given_conf(jnt_values=robot.rand_conf())
    robot.gen_meshmodel(rgb=rm.bc.red).attach_to(base)

    robot.goto_given_conf(jnt_values=robot.rand_conf())
    robot.gen_meshmodel(rgb=rm.bc.green).attach_to(base)

    robot.goto_given_conf(jnt_values=robot.rand_conf())
    robot.gen_meshmodel(rgb=rm.bc.blue).attach_to(base)
    base.run()
