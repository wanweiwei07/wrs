import math
import numpy as np
import wrs.robot_sim.manipulators.cobotta.cvrb1213 as cbta
import wrs.robot_sim.end_effectors.grippers.robotiq140.robotiq140 as rbtq1
import wrs.robot_sim.robots.single_arm_robot_interface as rsi


class CobottaPro1300WithRobotiq140(rsi.SglArmRobotInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name='cobotta_pro_1300',
                 enable_cc=True):
        super().__init__(pos, rotmat, name, enable_cc)
        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 4
        home_conf[2] = math.pi / 4
        home_conf[4] = math.pi / 2
        self.manipulator = cbta.CVRB1213(pos=self.pos, rotmat=self.rotmat, home_conf=home_conf,
                                         name=name + "_manipulator", enable_cc=enable_cc)
        self.end_effector = rbtq1.Robotiq140(pos=self.manipulator.gl_flange_pos,
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
        elb = self.cc.add_cce(self.end_effector.palm.lnk_list[0])
        elol0 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[0].lnk)
        elol1 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[1].lnk)
        elol2 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[2].lnk)
        elol3 = self.cc.add_cce(self.end_effector.lft_outer_jlc.jnts[3].lnk)
        elil0 = self.cc.add_cce(self.end_effector.lft_inner_jlc.jnts[0].lnk)
        erol0 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[0].lnk)
        erol1 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[1].lnk)
        erol2 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[2].lnk)
        erol3 = self.cc.add_cce(self.end_effector.rgt_outer_jlc.jnts[3].lnk)
        eril0 = self.cc.add_cce(self.end_effector.rgt_inner_jlc.jnts[0].lnk)
        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [ml4, ml5, elb, elol0, elol1, elol2, elol3, elil0, erol0, erol1, erol2, erol3, eril0]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)

        # For detection of collision with held work
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3, ml4]

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
    robot = CobottaPro1300WithRobotiq140(enable_cc=True)
    robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    base.run()
