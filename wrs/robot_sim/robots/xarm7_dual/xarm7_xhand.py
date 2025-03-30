import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.xarm7.xarm7 as manipulator
import wrs.robot_sim.end_effectors.multifinger.xhand.xhand_right as end_effector


class XArm7XHR(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="xarm7", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        self.manipulator = manipulator.XArm7(pos=self.pos, rotmat=self.rotmat, name=name + "_manipulator",
                                             home_conf=rm.vec(rm.pi/6, rm.pi/6, rm.pi/12, rm.pi/3, rm.pi, rm.pi/3, -rm.pi/2),
                                             enable_cc=False)
        self.end_effector = end_effector.XHandRight(pos=self.manipulator.gl_flange_pos,
                                                    rotmat=self.manipulator.gl_flange_rotmat,
                                                    coupling_offset_pos=rm.zeros(3),
                                                    coupling_offset_rotmat=rm.rotmat_from_euler(0,0,rm.pi),
                                                    name=name + "_eef")
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # end_effector
        ee_cces = []
        for id, cdlnk in enumerate(self.end_effector.cdelements):
            if id != 5 and id != 10:
                ee_cces.append(self.cc.add_cce(cdlnk))
        # manipulator
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = ee_cces + [ml4, ml5]
        into_list = [ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces[1:]

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
    from wrs import wd, mgm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = XArm7XHR(enable_cc=True)
    # robot.change_jaw_width(.05)
    # robot.cc.show_cdprim()
    robot.goto_given_conf(jnt_values=np.radians(np.array([20, -90, 120, 30, 0, 40, 0])))
    robot.cc.show_cdprim()
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)

    tgt_pos = np.array([.3, .3, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
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
