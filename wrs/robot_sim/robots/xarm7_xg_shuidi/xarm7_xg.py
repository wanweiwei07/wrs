import math
import numpy as np
from wrs import robot_sim as xg, robot_sim as x7, robot_sim as ri, modeling as mgm


class XArm7XG(ri.SglArmRobotInterface):
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
        # ee
        eb = self.cc.add_cce(self.end_effector.palm.lnk_list[0])
        el0 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[0].lnk)
        el1 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[1].lnk)
        er0 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[0].lnk)
        er1 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[1].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0], toggle_extcd=False)
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        ml6 = self.cc.add_cce(self.manipulator.jlc.jnts[6].lnk)
        from_list = [ml4, ml5, ml6, eb, el0, el1, er0, er1]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def fix_to(self, pos, rotmat, jnt_values=None):
        self.pos = pos
        self.rotmat = rotmat
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
