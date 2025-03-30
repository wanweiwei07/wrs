import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.agv.agv_interface as ai
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc


class A2WAGV(ai.AGVInterface):

    def __init__(self, home_pos=np.zeros(3), home_rotmat=np.eye(3), name="A2WAGV", enable_cc=False):
        """
        2D collision detection when enable_cc is False
        :param pos:
        :param rotmat:
        :param name:
        :param enable_cc:
        """
        current_file_dir = os.path.dirname(__file__)
        super().__init__(home_pos=home_pos, home_rotmat=home_rotmat, name=name, enable_cc=enable_cc)
        # anchor
        self.anchor = rkjlc.rkjl.Anchor(name=name + "_anchor", pos=self.movement_jlc.gl_flange_pos,
                                        rotmat=self.movement_jlc.gl_flange_rotmat, n_flange=1, n_lnk=9)
        # anchor flangz
        self.anchor.loc_flange_pose_list[0] = [np.array([.0, .0, .0]), np.eye(3)]
        # anchor body
        self.anchor.lnk_list[0].name = name + "_body"
        self.anchor.lnk_list[0].loc_pos = np.array([.0, .0, .0])
        self.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base_link.stl"), name=name + "_body")
        # anchor wheel1
        ## anchor wheel1 yaw
        self.anchor.lnk_list[1].name = name + "_wheel1_yaw"
        self.anchor.lnk_list[1].loc_pos = np.array([.24, -0.175, 0.0365])
        self.anchor.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_1_yaw.stl"), name=name + "_wheel1yaw")
        self.anchor.lnk_list[1].cmodel.rgba = rm.const.tab20_list[14]
        ## anchor wheel1
        self.anchor.lnk_list[2].name = name + "_wheel1"
        self.anchor.lnk_list[2].loc_pos = np.array([.24, -0.175, 0.0365])
        self.anchor.lnk_list[2].loc_rotmat = rm.rotmat_from_euler(-rm.pi / 2, 0, 0)
        self.anchor.lnk_list[2].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_1.stl"), name=name + "_wheel1")
        self.anchor.lnk_list[2].cmodel.rgba = rm.const.tab20_list[14]
        # anchor wheel2
        ## anchor wheel2 yaw
        self.anchor.lnk_list[3].name = name + "_wheel2_yaw"
        self.anchor.lnk_list[3].loc_pos = np.array([-.24, -0.175, 0.0365])
        self.anchor.lnk_list[3].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_2_yaw.stl"), name=name + "_wheel2yaw")
        self.anchor.lnk_list[3].cmodel.rgba = rm.const.tab20_list[14]
        ## anchor wheel2
        self.anchor.lnk_list[4].name = name + "_wheel2"
        self.anchor.lnk_list[4].loc_pos = np.array([-.24 - 0.01037, -0.175, 0.0365])
        self.anchor.lnk_list[4].loc_rotmat = rm.rotmat_from_euler(-rm.pi / 2, 0, 0)
        self.anchor.lnk_list[4].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_2.stl"), name=name + "_wheel2")
        self.anchor.lnk_list[4].cmodel.rgba = rm.const.tab20_list[14]
        # anchor wheel3
        ## anchor wheel3 yaw
        self.anchor.lnk_list[5].name = name + "_wheel3_yaw"
        self.anchor.lnk_list[5].loc_pos = np.array([-.24, 0.175, 0.0365])
        self.anchor.lnk_list[5].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_3_yaw.stl"), name=name + "_wheel3yaw")
        self.anchor.lnk_list[5].cmodel.rgba = rm.const.tab20_list[14]
        ## anchor wheel3
        self.anchor.lnk_list[6].name = name + "_wheel3"
        self.anchor.lnk_list[6].loc_pos = np.array([-.24 - 0.01037, 0.175, 0.0365])
        self.anchor.lnk_list[6].loc_rotmat = rm.rotmat_from_euler(rm.pi / 2, 0, 0)
        self.anchor.lnk_list[6].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_3.stl"), name=name + "_wheel3")
        self.anchor.lnk_list[6].cmodel.rgba = rm.const.tab20_list[14]
        # anchor wheel4
        ## anchor wheel4 yaw
        self.anchor.lnk_list[7].name = name + "_wheel4_yaw"
        self.anchor.lnk_list[7].loc_pos = np.array([.24, 0.175, 0.0365])
        self.anchor.lnk_list[7].loc_rotmat = rm.rotmat_from_euler(0, 0, rm.pi)
        self.anchor.lnk_list[7].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_4_yaw.stl"), name=name + "_wheel4yaw")
        self.anchor.lnk_list[7].cmodel.rgba = rm.const.tab20_list[14]
        ## anchor wheel4
        self.anchor.lnk_list[8].name = name + "_wheel4"
        self.anchor.lnk_list[8].loc_pos = np.array([.24, 0.175, 0.0365])
        self.anchor.lnk_list[8].loc_rotmat = rm.rotmat_from_euler(rm.pi / 2, 0, 0)
        self.anchor.lnk_list[8].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wheel_4.stl"), name=name + "_wheel4")
        self.anchor.lnk_list[8].cmodel.rgba = rm.const.tab20_list[14]
        if enable_cc:
            self.setup_cc()
        # backup
        self.jnt_values_bk = []

    @property
    def gl_flange_pose_list(self):
        return self.anchor.gl_flange_pose_list

    def setup_cc(self):
        agv = self.cc.add_cce(self.anchor.lnk_list[0])
        self.cc.enable_extcd_by_id_list(id_list=[agv], type="from")

    def restore_state(self):
        super().restore_state()
        self.anchor.pos = self.movement_jlc.gl_flange_pos
        self.anchor.rotmat = self.movement_jlc.gl_flange_rotmat

    def goto_given_conf(self, conf=np.zeros(3)):
        super().goto_given_conf(conf)
        self.anchor.pos = self.movement_jlc.gl_flange_pos
        self.anchor.rotmat = self.movement_jlc.gl_flange_rotmat

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='a2wagv_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.movement_jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
                                         toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        self.anchor.gen_stickmodel(toggle_root_frame=False, toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col

    def gen_meshmodel(self,
                      rgb=None,
                      alpha=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frames=False,
                      toggle_flange_frame=False,
                      toggle_cdprim=False,
                      toggle_cdmesh=False,
                      name='a2wagv_meshmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.anchor.gen_meshmodel(name + "_anchor", rgb=rgb,
                                  alpha=alpha, toggle_cdmesh=toggle_cdmesh, toggle_cdprim=toggle_cdprim,
                                  toggle_root_frame=toggle_jnt_frames,
                                  toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = A2WAGV(home_pos=rm.vec(.2, .3, 0), home_rotmat=rm.rotmat_from_axangle(ax=rm.const.z_ax, angle=rm.pi / 12),
                   enable_cc=False)
    # robot.goto_given_conf(conf=rm.vec(1,1,rm.pi/3))
    robot.gen_stickmodel().attach_to(base)
    robot.gen_meshmodel(alpha=1).attach_to(base)
    base.run()
