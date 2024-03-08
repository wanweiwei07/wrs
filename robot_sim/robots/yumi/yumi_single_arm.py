import math
import numpy as np
import robot_sim.manipulators.irb14050.irb14050 as irb14050
import robot_sim.end_effectors.gripper.yumi_gripper.yumi_gripper as yg
import robot_sim.robots.single_arm_robot_interface as ri


class YumiSglArm(ri.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="sglarm_yumi", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = irb14050.IRB14050(pos=self.pos, rotmat=self.rotmat, home_conf=np.zeros(7),
                                             name="irb14050_" + name, enable_cc=False)
        self.end_effector = yg.YumiGripper(pos=self.manipulator.gl_flange_pos,
                                           rotmat=self.manipulator.gl_flange_rotmat, name="yg_" + name)
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # TODO when pose is changed, oih info goes wrong
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        # manipulator
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, el0, el1, ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # TODO oiee?

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self._update_end_effector()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as mgm
    import modeling.collision_model as mcm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = YumiSglArm(enable_cc=True)
    robot.change_jaw_width(.05)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    print(jnt_values)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot.gen_meshmodel(alpha=.3, toggle_tcp_frame=True).attach_to(base)
    robot.show_cdprim()
    robot.unshow_cdprim()
    base.run()

    robot.goto_given_conf(jnt_values=np.array([0, np.pi / 2, np.pi * 11 / 20, 0, np.pi / 2, 0]))
    robot.show_cdprim()

    box = mcm.gen_box(xyz_lengths=np.array([0.1, .1, .1]), pos=tgt_pos, rgba=np.array([1, 1, 0, .3]))
    box.attach_to(base)
    tic = time.time()
    result, contacts = robot.is_collided(obstacle_list=[box], toggle_contacts=True)
    print(result)
    toc = time.time()
    print(toc - tic)
    for pnt in contacts:
        mgm.gen_sphere(pnt).attach_to(base)

    base.run()
