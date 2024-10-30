import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.xarm7.xarm7 as x7
import wrs.robot_sim.end_effectors.grippers.xarm_gripper.xarm_gripper as xg
import wrs.modeling.geometric_model as mgm


class XArm7XG(sari.SglArmRobotInterface):
    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm7_xg', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        home_conf = np.zeros(7)
        home_conf[1] = -math.pi / 6
        home_conf[3] = math.pi / 6
        self.manipulator = x7.XArm7(pos=self.pos, rotmat=self.rotmat, home_conf=home_conf,
                                    name=name + "_manipulator", enable_cc=enable_cc)
        self.end_effector = xg.XArmGripper(pos=self.manipulator.gl_flange_pos,
                                           rotmat=self.manipulator.gl_flange_rotmat, name=name + "_eef")
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_rotmat = self.end_effector.loc_acting_center_rotmat
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
        ml6 = self.cc.add_cce(self.manipulator.jlc.jnts[6].lnk)
        from_list = ee_cces + [ml4, ml5, ml6]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5, ml6], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces[1:]

    def fix_to(self, pos, rotmat, jnt_values=None):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat, jnt_values=jnt_values)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = XArm7XG(enable_cc=True)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    base.run()
