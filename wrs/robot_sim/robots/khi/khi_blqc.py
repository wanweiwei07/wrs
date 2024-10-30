import math
import numpy as np
import wrs.robot_sim._kinematics.jl as rkjl
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.robots.single_arm_robot_interface as sari
import wrs.robot_sim.manipulators.rs007l.rs007l as manipulator


class KHI_BLQC(sari.SglArmRobotInterface):
    """
    author: weiwei
    date: 20230826toyonaka
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="khi_g", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        # arm
        self.manipulator = manipulator.RS007L(pos=pos,
                                              rotmat=rotmat,
                                              name='rs007l', enable_cc=False)
        # tool changer
        self.tool_changer = rkjl.Anchor(name=name + '_tool_changer', pos=self.manipulator.gl_flange_pos,
                                        rotmat=self.manipulator.gl_flange_rotmat, n_flange=1)
        self.tool_changer.loc_flange_pose_list[0][0] = np.array([0, 0, .0315])
        self.tool_changer.lnk_list[0].cmodel = mcm.gen_stick(self.tool_changer.pos,
                                                             # TODO: change to combined model, 20230806
                                                             self.tool_changer.gl_flange_pose_list[0][0],
                                                             radius=0.05,
                                                             # rgba=[.2, .2, .2, 1], rgb will be overwritten
                                                             type='rect',
                                                             n_sec=36)
        # tool
        self.end_effector = None
        # tool center point
        self.manipulator.loc_tcp_pos = self.tool_changer.loc_flange_pose_list[0][0]
        self.manipulator.loc_tcp_rotmat = self.tool_changer.loc_flange_pose_list[0][1]
        # collision detection
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        # tc
        tc0 = self.cc.add_cce(self.tool_changer.lnk_list[0])
        # end_effector
        ee_cces = []
        if self.end_effector is not None:
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
        from_list = ee_cces + [tc0, ml3, ml4, ml5]
        into_list = [mlb, ml0]
        self.cc.set_cdpair_by_ids(from_list, into_list)
        # ext and inner
        self.cc.enable_extcd_by_id_list(id_list=[ml0, ml1, ml2, ml3, ml4, ml5] + ee_cces, type="from")
        self.cc.enable_innercd_by_id_list(id_list=[mlb, ml0, ml1, ml2, ml3], type="into")
        self.cc.dynamic_ext_list = ee_cces

    def attach_tool(self, tool):
        self.end_effector = tool
        if self.cc is not None:  # udpate cc
            self.reset_cc()
            self.setup_cc()
        self.update_end_effector()
        rel_pos, rel_rotmat = rm.rel_pose(self.end_effector.loc_acting_center_pos,
                                          self.end_effector.loc_acting_center_rotmat,
                                          self.tool_changer.loc_flange_pose_list[0][0],
                                          self.tool_changer.loc_flange_pose_list[0][1])
        self.manipulator.loc_tcp_pos = rel_pos
        self.manipulator.loc_tcp_rotmat = rel_rotmat

    def detach_tool(self):
        tool = self.end_effector
        self.end_effector = None
        if self.cc is not None:  # udpate cc
            self.reset_cc()
            self.setup_cc()
        self.update_end_effector()
        self.manipulator.loc_tcp_pos = self.tool_changer.loc_flange_pose_list[0][0]
        self.manipulator.loc_tcp_rotmat = self.tool_changer.loc_flange_pose_list[0][1]
        return tool

    def update_end_effector(self, ee_values=None):
        self.tool_changer.fix_to(pos=self._manipulator.gl_flange_pos, rotmat=self._manipulator.gl_flange_rotmat)
        if self.end_effector is not None:
            if ee_values is not None:
                self.end_effector.change_ee_values(ee_values=ee_values)
            self.end_effector.fix_to(pos=self.tool_changer.gl_flange_pose_list[0][0],
                                     rotmat=self.tool_changer.gl_flange_pose_list[0][1])

    def goto_given_conf(self, jnt_values, ee_values=None):
        result = self._manipulator.goto_given_conf(jnt_values=jnt_values)
        self.tool_changer.fix_to(pos=self._manipulator.gl_flange_pos, rotmat=self._manipulator.gl_flange_rotmat)
        self.update_end_effector(ee_values=ee_values)
        return result

    def fix_to(self, pos, rotmat):
        self._pos = pos
        self._rotmat = rotmat
        self.manipulator.fix_to(pos=pos, rotmat=rotmat)
        self.update_end_effector()

    def change_jaw_width(self, jaw_width):
        return self.change_ee_values(ee_values=jaw_width)

    def get_jaw_width(self):
        return self.get_ee_values()

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='khi_blqc_stickmodel'):
        m_col = super().gen_stickmodel(toggle_tcp_frame=toggle_tcp_frame,
                                       toggle_jnt_frames=toggle_jnt_frames,
                                       toggle_flange_frame=toggle_flange_frame,
                                       name=name)
        if self.tool_changer is not None:
            self.tool_changer.gen_stickmodel(toggle_root_frame=toggle_tcp_frame,
                                             toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='khi_blqc_meshmodel'):
        m_col = super().gen_meshmodel(rgb=rgb,
                                      alpha=alpha,
                                      toggle_tcp_frame=toggle_tcp_frame,
                                      toggle_jnt_frames=toggle_jnt_frames,
                                      toggle_flange_frame=toggle_flange_frame,
                                      toggle_cdprim=toggle_cdprim,
                                      toggle_cdmesh=toggle_cdmesh,
                                      name=name)
        if self.tool_changer is not None:
            print(self.tool_changer.pos)
            self.tool_changer.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_cdprim=toggle_cdprim,
                                            toggle_cdmesh=toggle_cdmesh).attach_to(m_col)
        return m_col


if __name__ == '__main__':
    import wrs.basis.robot_math as rm
    import wrs.modeling.geometric_model as mgm
    import wrs.visualization.panda.world as wd
    import wrs.robot_sim.end_effectors.grippers.or2fg7.or2fg7 as org
    import wrs.robot_sim.end_effectors.single_contact.screw_driver.orsd.orsd as ors
    from wrs import adp

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mgm.gen_frame().attach_to(base)
    # robot
    rbt = KHI_BLQC(enable_cc=True)
    rbt.gen_meshmodel().attach_to(base)
    # tool gripper
    ee_g_pos = np.array([-.2, .4, .19])
    ee_g_rotmat = rm.rotmat_from_euler(0, np.radians(180), 0)
    mgm.gen_frame(pos=ee_g_pos, rotmat=ee_g_rotmat).attach_to(base)
    ee_g = org.OR2FG7(pos=ee_g_pos,
                      rotmat=ee_g_rotmat,
                      coupling_offset_pos=np.array([0, 0, 0.0314]))
    ee_g_meshmodel = ee_g.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True)
    ee_g_meshmodel.attach_to(base)
    # tool screw driver
    ee_sd_pos = np.array([-.4, .6, .19])
    ee_sd_rotmat = rm.rotmat_from_euler(0, np.radians(180), np.radians(180))
    ee_sd = ors.ORSD(pos=ee_sd_pos,
                     rotmat=ee_sd_rotmat,
                     coupling_offset_pos=np.array([0, 0, 0.0314]))
    ee_sd_meshmodel = ee_sd.gen_meshmodel(toggle_jnt_frames=True, toggle_tcp_frame=True)
    ee_sd_meshmodel.attach_to(base)
    # start conf
    start_conf = rbt.get_jnt_values()
    # ad planner
    ad_planner = adp.ADPlanner(robot=rbt)
    # attach eeg
    mot_attach_eeg = ad_planner.gen_approach(goal_tcp_pos=ee_g_pos,
                                             goal_tcp_rotmat=ee_g_rotmat,
                                             start_jnt_values=start_conf,
                                             linear_direction=-rm.const.z_ax)
    rbt.goto_given_conf(mot_attach_eeg.jv_list[-1])
    rbt.attach_tool(ee_g)
    # eeg action
    ee_g_act_pos = np.array([.4, -.4, .19])
    ee_g_act_rotmat = ee_g_rotmat
    mot_eeg = ad_planner.gen_depart_approach_with_given_conf(
        start_jnt_values=mot_attach_eeg.jv_list[-1],
        end_jnt_values=mot_attach_eeg.jv_list[-1],
        depart_direction=-rm.const.x_ax,
        depart_distance=.3,
        approach_direction=rm.const.x_ax,
        approach_distance=.3)
    tool = rbt.detach_tool()
    # attach eesd
    end_jnt_values = rbt.ik(ee_sd_pos, ee_sd_rotmat)
    mot_attach_eesd = ad_planner.gen_depart_approach_with_given_conf(
        start_jnt_values=mot_attach_eeg.jv_list[-1],
        end_jnt_values=end_jnt_values,
        depart_direction=rm.const.z_ax,
        approach_direction=-rm.const.z_ax)
    rbt.goto_given_conf(mot_attach_eesd.jv_list[-1])
    rbt.attach_tool(ee_sd)
    # eesd action
    ee_sd_act_pos = np.array([.4, -.4, .19])
    ee_sd_act_rotmat = ee_sd_rotmat
    mot_eesd = ad_planner.gen_depart_approach_with_given_conf(
        start_jnt_values=mot_attach_eesd.jv_list[-1],
        end_jnt_values=mot_attach_eesd.jv_list[-1],
        depart_direction=rm.const.x_ax,
        depart_distance=.3,
        approach_direction=-rm.const.x_ax,
        approach_distance=.3)
    tool = rbt.detach_tool()


    class Data(object):
        def __init__(self, mot_data):
            self.counter = 0
            self.mot_data = mot_data


    print(mot_attach_eeg.robot, mot_eeg.robot, mot_attach_eesd.robot, mot_eesd.robot)

    anime_data = Data(mot_attach_eeg + mot_eeg + mot_attach_eesd + mot_eesd)


    def update(anime_data, task):
        if anime_data.counter > 0:
            anime_data.mot_data.mesh_list[anime_data.counter - 1].detach()
        if anime_data.counter >= len(anime_data.mot_data):
            # for mesh_model in anime_data.mot_data.mesh_list:
            #     mesh_model.detach()
            anime_data.counter = 0
        mesh_model = anime_data.mot_data.mesh_list[anime_data.counter]
        mesh_model.attach_to(base)
        mesh_model.show_cdprim()
        if base.inputmgr.keymap['space']:
            anime_data.counter += 1
        return task.again


    taskMgr.doMethodLater(0.01, update, "update",
                          extraArgs=[anime_data],
                          appendTask=True)

    base.run()
