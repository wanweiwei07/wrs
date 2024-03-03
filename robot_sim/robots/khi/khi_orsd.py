import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.rs007l.rs007l as manipulator
import robot_sim.end_effectors.single_contact.screw_driver.orsd.orsd as end_effector
import robot_sim.robots.single_arm_robot_interface as ai


class KHI_ORSD(ai.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="khi_g", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        # arm
        self.manipulator = manipulator.RS007L(pos=pos,
                                              rotmat=rotmat,
                                              homeconf=np.zeros(6),
                                              name='rs007l', enable_cc=False)
        # gripper
        self.end_effector = end_effector.ORSD(pos=self.manipulator.jnts[-1]['gl_posq'],
                                              rotmat=self.manipulator.jnts[-1]['gl_rotmatq'],
                                              coupling_offset_pos=np.array([0, 0, 0.0639]),
                                              name='orsd', enable_cc=False)
        # tool center point
        self.manipulator.jlc.flange_jnt_id = -1
        self.manipulator.jlc._loc_flange_pos = self.end_effector.action_center_pos
        self.manipulator.jlc._loc_flange_rotmat = self.end_effector.action_center_rotmat
        # collision detection
        if enable_cc:
            self.enable_cc()

    def _update_oof(self):
        """
        oof = object on flange
        this function is to be implemented by subclasses for updating ft-sensors, tool changers, end_type-effectors, etc.
        :return:
        author: weiwei
        date: 20230807
        """
        self.end_effector.fix_to(pos=self.manipulator.jnts[-1]['gl_posq'],
                                 rotmat=self.manipulator.jnts[-1]['gl_rotmatq'])

    def enable_cc(self):
        pass
        # TODO when pose is changed, oih info goes wrong
        # super().enable_cc()
        # self.cc.add_cdlnks(self.base_plate, [0])
        # self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        # self.cc.add_cdlnks(self.hnd.jlc, [0, 1, 2])
        # activelist = [self.base_plate.lnks[0],
        #               self.arm.lnks[0],
        #               self.arm.lnks[1],
        #               self.arm.lnks[2],
        #               self.arm.lnks[3],
        #               self.arm.lnks[4],
        #               self.arm.lnks[5],
        #               self.arm.lnks[6],
        #               self.hnd.jlc.lnks[0],
        #               self.hnd.jlc.lnks[1],
        #               self.hnd.jlc.lnks[2]]
        # self.cc.set_active_cdlnks(activelist)
        # fromlist = [self.base_plate.lnks[0],
        #             self.arm.lnks[0],
        #             self.arm.lnks[1]]
        # intolist = [self.arm.lnks[3]]
        # self.cc.set_cdpair(fromlist, intolist)
        # fromlist = [self.base_plate.lnks[0],
        #             self.arm.lnks[1]]
        # intolist = [self.hnd.jlc.lnks[0],
        #             self.hnd.jlc.lnks[1],
        #             self.hnd.jlc.lnks[2]]
        # self.cc.set_cdpair(fromlist, intolist)
        # # TODO is the following update needed?
        # for oih_info in self.oih_infos:
        #     obj_cmodel = oih_info['collision_model']
        #     self.hold(obj_cmodel)

    def hold(self, hnd_name, objcm, jawwidth=None):
        """
        the obj_cmodel is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if jawwidth is not None:
            self.end_effector.jaw_to(jawwidth)
        rel_pos, rel_rotmat = self.cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        intolist = [self.arm.lnks[0],
                    self.arm.lnks[1],
                    self.arm.lnks[2],
                    self.arm.lnks[3],
                    self.arm.lnks[4]]  # TODO unchecked
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, intolist))
        return rel_pos, rel_rotmat

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='khi_sd_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.manipulator.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                        tcp_loc_pos=tcp_loc_pos,
                                        tcp_loc_rotmat=tcp_loc_rotmat,
                                        toggle_tcpcs=toggle_tcp_frame,
                                        toggle_jntscs=toggle_jnt_frames,
                                        toggle_connjnt=toggle_flange_frame).attach_to(stickmodel)
        self.end_effector.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frames).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      rgba=None,
                      name='khi_sd_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.manipulator.gen_meshmodel(toggle_tcp_frame=toggle_tcp_frame, toggle_jnt_frames=toggle_jnt_frames,
                                       rgba=rgba).attach_to(meshmodel)
        self.end_effector.gen_meshmodel(toggle_tcp_frame=False,
                                        toggle_jnt_frames=toggle_jnt_frames,
                                        rgba=rgba).attach_to(meshmodel)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(meshmodel)
        return meshmodel

    def ik(self,
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           tcp_loc_pos=None,
           tcp_loc_rotmat=None):
        return self.manipulator.analytical_ik(tgt_pos,
                                              tgt_rotmat,
                                              tcp_loc_pos=tcp_loc_pos,
                                              tcp_loc_rotmat=tcp_loc_rotmat)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

    gm.gen_frame().attach_to(base)
    robot_s = KHI_ORSD(enable_cc=True)
    # robot_s.jaw_to(.02)
    robot_s.gen_meshmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # robot_s.gen_meshmodel(toggle_flange_frame=False, toggle_jnt_frames=False).attach_to(base)
    robot_s.gen_stickmodel(toggle_tcp_frame=True, toggle_jnt_frames=True).attach_to(base)
    # base.run()
    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    component_name = 'arm'
    jnt_values = robot_s.ik(tgt_pos, tgt_rotmat)
    robot_s.fk(jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcp_frame=True)
    robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
