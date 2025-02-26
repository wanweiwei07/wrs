import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.cobotta.cvr038 as cbta
import wrs.robot_sim.end_effectors.grippers.wrs_gripper.wrs_tweezer as wt


class CobottaTweezer(sari.SglArmRobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="cobotta_largehand", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        this_dir, this_filename = os.path.split(__file__)
        # base plate
        self.base_plate = rkjl.Anchor(name=name + "_base_plate", pos=self.pos, rotmat=self.rotmat, n_flange=1)
        self.base_plate.loc_flange_pose_list[0] = [np.array([0, 0, 0.035]), np.eye(3)]
        self.base_plate.lnk_list[0].cmodel = mcm.CollisionModel(os.path.join(this_dir, "meshes", "base_plate.stl"))
        self.base_plate.lnk_list[0].cmodel.rgb = rm.const.steel_gray
        # arm
        self.manipulator = cbta.CVR038(pos=self.base_plate.gl_flange_pose[0],
                                       rotmat=self.base_plate.gl_flange_pose[1],
                                       name='CobottaLarge_arm', enable_cc=False)
        # grippers
        self.end_effector = wt.WRSTweezer(pos=self.manipulator.gl_flange_pos,
                                          rotmat=self.manipulator.gl_flange_rotmat,
                                          name='CobottaTweezer_hnd')
        # tool center point
        self.manipulator.jlc.loc_tcp_pos = self.end_effector.loc_acting_center_pos
        self.manipulator.jlc.loc_tcp_rotmat = self.end_effector.loc_acting_center_rotmat
        # collision detection
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # ee
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
        from_list = ee_cces + [ml3, ml4, ml5]
        into_list = [mlb, ml0, ml1]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5], type="from")
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces[1:]

    def fix_to(self, pos, rotmat, wide=None):
        self._pos = pos
        self._rotmat = rotmat
        self.base_plate.fix_to(pos=pos, rotmat=rotmat)
        self.manipulator.fix_to(pos=self.base_plate.gl_flange_pose[0], rotmat=self.base_plate.gl_flange_pose[1])
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
                                      toggle_root_frame=toggle_jnt_frames,
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
    robot = CobottaTweezer(enable_cc=True)

    tgt_pos = np.array([.12, -.1, .1])
    tgt_rotmat = rm.rotmat_from_axangle([5, 1, 3], math.pi)
    jnt_values = robot.ik(tgt_pos, tgt_rotmat)
    if jnt_values is not None:
        robot.goto_given_conf(jnt_values=jnt_values)
        robot_meshmodel = robot.gen_meshmodel(toggle_tcp_frame=True)
        print(robot.is_collided(toggle_dbg=True))
        robot_meshmodel.attach_to(base)
    base.run()

    # current_jnt_values = robot_s.ik(tgt_pos, tgt_rotmat)
    # robot_s.fix_to(tgt_pos, tgt_rotmat)
    # current_jnt_values =np.array([ 1.68747252,  0.97073813,  2.2426744 ,  0.11973604, -1.63202317,
    #    -0.01484051])
    # fk_results = robot_s.goto_given_conf(jnt_values=current_jnt_values)
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
