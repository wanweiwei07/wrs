import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.robots.dual_arm_robot_interface as dari
import wrs.robot_sim.agv.a2w_agv.a2w_agv as agv
import wrs.robot_sim.manipulators.a2w_arm.a2w_arm as manipulator
import wrs.robot_sim.end_effectors.grippers.a2w_gripper.a2w_gripper as end_effector


class A2W(dari.DualArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="a2w", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # the body anchor
        self._body = rkjlc.rkjl.Anchor(name=self.name + "_anchor", pos=self.pos,
                                       rotmat=self.rotmat, n_flange=2, n_lnk=1)
        self._body.loc_flange_pose_list[0] = [rm.vec(0, .016, .22),
                                              rm.rotmat_from_euler(-rm.pi / 2.0, 0, rm.pi)]
        self._body.loc_flange_pose_list[1] = [rm.vec(0, -.016, .22),
                                              rm.rotmat_from_euler(-rm.pi / 2.0, 0, 0)]
        self._body.lnk_list[0].name = "a2w_dual_base_link"
        self._body.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "arm_base_link.stl"), name=self.name + "_body")
        self._body.lnk_list[0].cmodel.rgba = rm.const.hug_gray
        # left arm
        self._lft_arm = manipulator.A2WLeftArm(pos=self._body.gl_flange_pose_list[0][0],
                                                rotmat=self._body.gl_flange_pose_list[0][1],
                                                name=self.name + "_left_arm", enable_cc=False)
        # right arm
        self._rgt_arm = manipulator.A2WRightArm(pos=self._body.gl_flange_pose_list[1][0],
                                                  rotmat=self._body.gl_flange_pose_list[1][1],
                                                  name=self.name + "_right_arm", enable_cc=False)
        # self.agv = agv.A2WAGV(home_pos=self.pos, home_rotmat=self.rotmat, name=name + "_agv", enable_cc=False)
        # self._body = rkjlc.JLChain(pos=self.agv.gl_flange_pose_list[0][0],
        #                           rotmat=self.agv.gl_flange_pose_list[0][1],
        #                           n_dof=3, name=name + "_body")
        # # body lift
        # self._body.jnts[0].change_type(type=rkjlc.const.JntType.PRISMATIC, motion_range=rm.vec(-.285, .285))
        # self._body.jnts[0].loc_pos = rm.vec(-.039, .0, .8375)
        # self._body.jnts[0].loc_motionax = rm.const.z_ax
        # self._body.jnts[0].lnk.cmodel = mcm.CollisionModel(
        #     initor=os.path.join(current_file_dir, "meshes", "body_link1.stl"), name=name + "_body")
        # # body revolute
        # self._body.jnts[1].loc_pos = rm.vec(.1395, .0, .015)
        # self._body.jnts[1].loc_rotmat = rm.rotmat_from_euler(-rm.pi / 2, 0, 0)
        # self._body.jnts[1].motion_range = rm.vec(0, 2.3562)
        # self._body.jnts[1].loc_motionax = rm.const.z_ax
        # self._body.jnts[1].lnk.cmodel = mcm.CollisionModel(
        #     initor=os.path.join(current_file_dir, "meshes", "body_link2.stl"), name=name + "_body")
        # # head revolute
        # self._body.jnts[2].loc_pos = rm.vec(0.066, 0.0, 0.3465)
        # self._body.jnts[2].loc_rotmat = rm.rotmat_from_euler(-rm.pi / 2, 0, 0)
        # self._body.jnts[2].loc_motionax = rm.const.z_ax
        # self._body.jnts[2].lnk.cmodel = mcm.CollisionModel(
        #     initor=os.path.join(current_file_dir, "meshes", "head_link.stl"), name=name + "_body")

        # self.manipulator = manipulator.A2WLeftArm(pos=self.pos, rotmat=self.rotmat,
        #                                           ik_solver=None, name=name + "_manipulator", enable_cc=False)
        # self.end_effector = end_effector.A2WGripper(pos=self.manipulator.gl_flange_pos,
        #                                             rotmat=self.manipulator.gl_flange_rotmat,
        #                                             name=name + "_eef")
        # tool center point
        # self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        # self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # if self.cc is not None:
        #     self.setup_cc()

    # def setup_cc(self):
    #     # end_effector
    #     ee_cces = []
    #     for id, cdlnk in enumerate(self.end_effector.cdelements):
    #         if id != 5 and id != 10:
    #             ee_cces.append(self.cc.add_cce(cdlnk))
    #     # manipulator
    #     ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
    #     ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
    #     ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
    #     ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
    #     ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
    #     ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
    #     from_list = ee_cces + [ml4, ml5]
    #     into_list = [ml0, ml1]
    #     self.cc.set_cdpair_by_ids(from_list, into_list)
    #     # ext and inner
    #     self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
    #     self.cc.enable_innercd_by_id_list(id_list=[ml0, ml1, ml2, ml3], type="into")
    #     self.cc.dynamic_ext_list = ee_cces[1:]
    #
    # def fix_to(self, pos, rotmat):
    #     self._pos = pos
    #     self._rotmat = rotmat
    #     self.manipulator.fix_to(pos=pos, rotmat=rotmat)
    #     self.update_end_effector()

    def get_jaw_width(self):
        return self.end_effector.get_jaw_width()

    def change_jaw_width(self, jaw_width):
        self.end_effector.change_jaw_width(jaw_width=jaw_width)

    # def gen_stickmodel(self,
    #                    toggle_tcp_frame=False,
    #                    toggle_jnt_frames=False,
    #                    toggle_flange_frame=False):
    #     m_col = super().gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
    #                                    toggle_jnt_frames=toggle_jnt_frames,
    #                                    toggle_flange_frame=toggle_flange_frame)
    #     self.agv.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
    #                             toggle_jnt_frames=False,
    #                             toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
    #     return m_col
    #
    # def gen_meshmodel(self,
    #                   rgb=None,
    #                   alpha=1,
    #                   toggle_tcp_frame=False,
    #                   toggle_jnt_frames=False,
    #                   toggle_flange_frame=False,
    #                   toggle_cdprim=False,
    #                   toggle_cdmesh=False):
    #     m_col = super().gen_meshmodel(rgb=rgb,
    #                                   alpha=alpha,
    #                                   toggle_tcp_frame=toggle_tcp_frame,
    #                                   toggle_jnt_frames=toggle_jnt_frames,
    #                                   toggle_flange_frame=toggle_flange_frame,
    #                                   toggle_cdprim=toggle_cdprim,
    #                                   toggle_cdmesh=toggle_cdmesh)
    #     self.agv.gen_meshmodel(rgb=rgb,
    #                            alpha=alpha,
    #                            toggle_tcp_frame=toggle_tcp_frame,
    #                            toggle_jnt_frames=False,
    #                            toggle_flange_frame=toggle_flange_frame,
    #                            toggle_cdprim=toggle_cdprim,
    #                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
    #     return m_col


if __name__ == '__main__':
    from wrs import wd, mgm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    robot = A2W(enable_cc=True)
    robot.gen_meshmodel(alpha=1, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    #
    # tgt_pos = np.array([.3, .1, .3])
    # tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    # mgm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # jnt_values = robot.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat, toggle_dbg=False)
    # print("IK result: ", jnt_values)
    # if jnt_values is not None:
    #     robot.goto_given_conf(jnt_values=jnt_values)
    #     robot.gen_meshmodel(alpha=.5, toggle_tcp_frame=True, toggle_jnt_frames=False).attach_to(base)
    #     robot.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot.show_cdprim()
    # robot.unshow_cdprim()
    base.run()
