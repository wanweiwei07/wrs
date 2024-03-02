import os
import math
import numpy as np
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import robot_sim.end_effectors.gripper.cobotta_gripper.cobotta_gripper as cbtg
import robot_sim.robots.single_arm_robot_interface as ri


class Cobotta(ri.SglArmRbtInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)

        home_conf = np.zeros(6)
        home_conf[1] = -math.pi / 6
        home_conf[2] = math.pi / 2
        home_conf[4] = math.pi / 6
        self.manipulator = cbta.CobottaArm(pos=self.pos, rotmat=self.rotmat, home_conf=home_conf, name="cobotta_arm",
                                           enable_cc=False)
        self.end_effector = cbtg.CobottaGripper(pos=self.manipulator.gl_flange_pos,
                                                rotmat=self.manipulator.gl_flange_rotmat, name="cobotta_hnd",
                                                enable_cc=False)
        # tool center point
        self.manipulator.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # collision detection
        if enable_cc:
            self.enable_cc()

    # def enable_cc(self):
    #     # TODO when pose is changed, oih info goes wrong
    #     super().enable_cc()
    #     self.cc.add_cdlnks(self.base_plate, [0])
    #     self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
    #     self.cc.add_cdlnks(self.hnd.jlc, [0, 1, 2])
    #     activelist = [self.base_plate.lnks[0],
    #                   self.arm.lnks[0],
    #                   self.arm.lnks[1],
    #                   self.arm.lnks[2],
    #                   self.arm.lnks[3],
    #                   self.arm.lnks[4],
    #                   self.arm.lnks[5],
    #                   self.arm.lnks[6],
    #                   self.hnd.jlc.lnks[0],
    #                   self.hnd.jlc.lnks[1],
    #                   self.hnd.jlc.lnks[2]]
    #     self.cc.set_active_cdlnks(activelist)
    #     fromlist = [self.base_plate.lnks[0],
    #                 self.arm.lnks[0],
    #                 self.arm.lnks[1]]
    #     intolist = [self.arm.lnks[3]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     fromlist = [self.base_plate.lnks[0],
    #                 self.arm.lnks[1]]
    #     intolist = [self.hnd.jlc.lnks[0],
    #                 self.hnd.jlc.lnks[1],
    #                 self.hnd.jlc.lnks[2]]
    #     self.cc.set_cdpair(fromlist, intolist)
    #     # TODO is the following update needed?
    #     for oih_info in self.oih_infos:
    #         objcm = oih_info['collision_model']
    #         self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self._update_end_effector()
        # self.end_effector.fix_to(pos=self.manipulator.gl_flange_pos, rotmat=self.manipulator.gl_flange_rotmat)

    # def hold(self, hnd_name, objcm, jawwidth=None):
    #     """
    #     the objcm is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param objcm:
    #     :return:
    #     """
    #     if hnd_name not in self.hnd_dict:
    #         raise ValueError("Hand name does not exist!")
    #     if jawwidth is not None:
    #         self.hnd_dict[hnd_name].change_jaw_width(jawwidth)
    #     rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_pose_to_tcp(objcm.get_pos(), objcm.get_rotmat())
    #     intolist = [self.arm.lnks[0],
    #                 self.arm.lnks[1],
    #                 self.arm.lnks[2],
    #                 self.arm.lnks[3],
    #                 self.arm.lnks[4]]
    #     self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
    #     return rel_pos, rel_rotmat

    # def get_oih_list(self):
    #     return_list = []
    #     for obj_info in self.oih_infos:
    #         objcm = obj_info['collision_model']
    #         objcm.set_pos(obj_info['gl_pos'])
    #         objcm.set_rotmat(obj_info['gl_rotmat'])
    #         return_list.append(objcm)
    #     return return_list
    #
    # def release(self, hnd_name, objcm, jawwidth=None):
    #     """
    #     the objcm is added as a part of the robot_s to the cd checker
    #     :param jawwidth:
    #     :param objcm:
    #     :return:
    #     """
    #     if hnd_name not in self.hnd_dict:
    #         raise ValueError("Hand name does not exist!")
    #     if jawwidth is not None:
    #         self.hnd_dict[hnd_name].change_jaw_width(jawwidth)
    #     for obj_info in self.oih_infos:
    #         if obj_info['collision_model'] is objcm:
    #             self.cc.delete_cdobj(obj_info)
    #             self.oih_infos.remove(obj_info)
    #             break

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='single_arm_robot_interface_stickmodel'):
        m_col = mc.ModelCollection(name=name)
        self.manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.end_effector.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                         toggle_jnt_frames=toggle_jnt_frames).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='single_arm_robot_interface_meshmodel'):
        m_col = mc.ModelCollection(name=name)
        self.manipulator.gen_meshmodel(rgb=rgb,
                                       alpha=alpha,
                                       toggle_tcp_frame=toggle_tcp_frame,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       toggle_cdprim=toggle_cdprim,
                                       toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        self.end_effector.gen_meshmodel(rgb=rgb,
                                        alpha=alpha,
                                        toggle_tcp_frame=toggle_tcp_frame,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        toggle_cdprim=toggle_cdprim,
                                        toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

    gm.gen_frame().attach_to(base)
    robot_s = Cobotta(enable_cc=False)
    # robot_s.jaw_to(.02)
    robot_s.gen_meshmodel(alpha=.5, toggle_tcp_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot_s.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # base.run()
    tgt_pos = np.array([.3, .1, .3])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    jnt_values = robot_s.ik(tgt_pos=tgt_pos, tgt_rotmat=tgt_rotmat)
    if jnt_values is not None:
        robot_s.goto_given_conf(jnt_values=jnt_values)
        robot_s.gen_meshmodel(toggle_tcp_frame=True).attach_to(base)
    base.run()
    # robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
    tic = time.time()
    result = robot_s.is_collided()
    toc = time.time()
    print(result, toc - tic)
    base.run()
