import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.cobotta_arm.cobotta_arm as cbta
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_gripper_v4 as wg4


class CobottaLarge(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta_largehand", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_plate = rkjlc.JLChain(pos=pos,
                                     rotmat=rotmat,
                                     n_dof=1,
                                     name='base_plate_ripps')
        self.base_plate.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "base_plate.stl"))
        self.base_plate.jnts[0].lnk.loc_pos = np.array([0, 0, -0.035])
        self.base_plate.jnts[0].lnk.cmodel.rgba = np.array([.55, .55, .55, 1])
        # self.base_plate.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "base_plate_I301.stl"))
        # self.base_plate.jnts[0].lnk.loc_pos = np.array([0, 0, -0.01])
        # self.base_plate.jnts[0].lnk.cmodel.rgba = np.array([.55, .55, .55, 1])
        self.base_plate.finalize()
        # arm
        self.manipulator = cbta.CobottaArm(pos=self.base_plate.gl_flange_pos,
                                           rotmat=self.base_plate.gl_flange_rotmat,
                                           name='CobottaLarge_arm', enable_cc=False)
        # grippers
        self.end_effector = wg4.WRSGripper4(pos=self.manipulator.gl_flange_pos,
                                              rotmat=self.manipulator.gl_tcp_rotmat,
                                              name='CobottaLarge_hnd')
        # tool center point
        self.manipulator.jlc.tcp_jnt_id = -1
        self.manipulator.jlc.tcp_loc_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.jlc.tcp_loc_rotmat = self.end_effector.loc_acting_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # ee
        elb = self.cc.add_cce(self.end_effector.jlc.anchor.lnk_list[0])
        ell0 = self.cc.add_cce(self.end_effector.jlc.jnts[0].lnk)
        ell1 = self.cc.add_cce(self.end_effector.jlc.jnts[1].lnk)
        ell2 = self.cc.add_cce(self.end_effector.jlc.jnts[2].lnk)
        elr0 = self.cc.add_cce(self.end_effector.jlc.jnts[3].lnk)
        elr1 = self.cc.add_cce(self.end_effector.jlc.jnts[4].lnk)
        elr2 = self.cc.add_cce(self.end_effector.jlc.jnts[5].lnk)

        # manipulator
        mlb = self.cc.add_cce(self.manipulator.jlc.anchor.lnk_list[0])
        ml0 = self.cc.add_cce(self.manipulator.jlc.jnts[0].lnk)
        ml1 = self.cc.add_cce(self.manipulator.jlc.jnts[1].lnk)
        ml2 = self.cc.add_cce(self.manipulator.jlc.jnts[2].lnk)
        ml3 = self.cc.add_cce(self.manipulator.jlc.jnts[3].lnk)
        ml4 = self.cc.add_cce(self.manipulator.jlc.jnts[4].lnk)
        ml5 = self.cc.add_cce(self.manipulator.jlc.jnts[5].lnk)
        from_list = [elb, ell2, elr2, ml3, ml4, ml5]
        into_list = [mlb, ml0, ml1, ml2]
        self.cc.set_cdpair_by_ids(from_list, into_list)

    def fix_to(self, pos, rotmat, wide=None):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector(wide)

    def fk(self, jnt_values, gripper_wide=None, toggle_jacobian=False, update=True):
        gl_flange_pos, gl_flange_rotmat = self._manipulator.fk(jnt_values=jnt_values, toggle_jacobian=toggle_jacobian,
                                                               update=update)
        self.update_end_effector(gripper_wide)
        ee_jaw_center = self.manipulator.jlc.tcp_loc_pos
        gl_tcp_pos = gl_flange_pos + gl_flange_rotmat.dot(ee_jaw_center)
        gl_tcp_rotmat = gl_flange_rotmat
        return (gl_tcp_pos, gl_tcp_rotmat)

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='cobotta_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.base_plate.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self._manipulator is not None:
            self._manipulator.gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                             toggle_jnt_frames=toggle_jnt_frames,
                                             toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        if self.end_effector is not None:
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
                      name='cobotta_largehand_rack_meshmodel'):
        """

        :param tcp_jnt_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param toggle_tcpcs:
        :param toggle_jntscs:
        :param rgba:
        :param name:
        :return:
        """
        m_col = mmc.ModelCollection(name=name)
        self.base_plate.gen_meshmodel(rgb=rgb,
                                      alpha=alpha,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_flange_frame,
                                      toggle_cdprim=toggle_cdprim,
                                      toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self._manipulator is not None:
            self.manipulator.gen_meshmodel(rgb=rgb,
                                           alpha=alpha,
                                           toggle_tcp_frame=False,
                                           toggle_jnt_frames=toggle_jnt_frames,
                                           toggle_flange_frame=toggle_flange_frame,
                                           toggle_cdprim=toggle_cdprim,
                                           toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        if self._end_effector is not None:
            self.end_effector.gen_meshmodel(rgb=rgb,
                                            alpha=alpha,
                                            toggle_tcp_frame=toggle_tcp_frame,
                                            toggle_jnt_frames=toggle_jnt_frames,
                                            toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import wrs.basis.robot_math as rm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot_s = CobottaLarge(enable_cc=True)

    tgt_pos = np.array([.25, .2, .15])
    tgt_rotmat = rm.rotmat_from_axangle([5, 1, 3], math.pi * 2 / 3)

    # current_jnt_values = robot_s.ik(tgt_pos, tgt_rotmat)
    robot_s.fix_to(tgt_pos,tgt_rotmat)
    # # current_jnt_values =np.array([ 1.68747252,  0.97073813,  2.2426744 ,  0.11973604, -1.63202317,
    # #    -0.01484051])
    # fk_results = robot_s.fk(jnt_values=current_jnt_values)
    # print(fk_results)
    # # gm.gen_frame(*fk_results).attach_to(base)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcp_frame=True)
    robot_s_meshmodel.attach_to(base)
    # # robot_s.show_cdprimit()
    # # robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()