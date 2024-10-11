import math
import numpy as np
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.rs007l.rs007l as manipulator
import wrs.robot_sim.end_effectors.grippers.or2fg7.or2fg7 as end_effector


class KHI_OR2FG7(sari.SglArmRobotInterface):

    def __init__(self,
                 pos=np.zeros(3),
                 rotmat=np.eye(3),
                 name="khi_or2fg7",
                 enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        # manipulator
        self.manipulator = manipulator.RS007L(pos=pos,
                                              rotmat=rotmat,
                                              name='rs007l',
                                              enable_cc=False)
        # end_effector
        self.end_effector = end_effector.OR2FG7(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat,
                                                name='or2fg7')
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
        from_list = [elb, el0, el1, ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        self.cc.dynamic_into_list = [mlb, ml0, ml1, ml2, ml3]
        self.cc.dynamic_ext_list = [elb, el0, el1]

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
    import wrs.visualization.panda.world as wd
    import wrs.basis.robot_math as rm
    import wrs.modeling.geometric_model as mgm

    base = wd.World(cam_pos=[2, 2, 3], lookat_pos=[0, 0, .5])

    mgm.gen_frame().attach_to(base)
    robot = KHI_OR2FG7(pos=np.array([0,0,0.2]), enable_cc=True)
    robot.change_jaw_width(.02)
    # robot.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot_s.gen_meshmodel(toggle_flange_frame=False, toggle_jnt_frames=False).attach_to(base)
    # robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # base.run()
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, ax_length=1).attach_to(base)
    # base.run()
    jnt_values = robot.ik(tgt_pos, tgt_rotmat)
    print(jnt_values)
    robot.goto_given_conf(jnt_values=jnt_values)
    mesh_model = robot.gen_meshmodel(toggle_tcp_frame=True)
    mesh_model.attach_to(base)
    # robot_s.show_cdprimit()
    # robot.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
