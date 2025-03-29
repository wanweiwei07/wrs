import os
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.agv.agv_interface as ai
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc


class Shuidi(ai.AGVInterface):

    def __init__(self, home_pos=np.zeros(3), home_rotmat=np.eye(3), name="Shuidi", enable_cc=False):
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
                                        rotmat=self.movement_jlc.gl_flange_rotmat, n_flange=1, n_lnk=3)
        # anchor flange
        self.anchor.loc_flange_pose_list[0] = [np.array([.0, .0, .445862]), np.eye(3)]
        # anchor body
        self.anchor.lnk_list[0].name = name + "_body"
        self.anchor.lnk_list[0].loc_pos = np.array([.0, .0, .0])
        self.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "shuidi_agv.stl"), name="shuidi_body")
        # anchor battery
        self.anchor.lnk_list[1].name = name + "_battery"
        self.anchor.lnk_list[1].loc_pos = np.array([.0, .0, .277])
        self.anchor.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "battery.stl"), name="shuidi_battery")
        self.anchor.lnk_list[1].cmodel.rgba = rm.const.tab20_list[14]
        # anchor battery fixture
        self.anchor.lnk_list[2].name = name + "_battery_fixture"
        self.anchor.lnk_list[2].loc_pos = np.array([.0, .0, .277])
        self.anchor.lnk_list[2].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "battery_fixture.stl"), name="shuidi_battery_fixture")
        self.anchor.lnk_list[2].cmodel.rgba = rm.const.tab20_list[15]
        if enable_cc:
            self.setup_cc()
        # backup
        self.jnt_values_bk = []

    @property
    def gl_flange_pose_list(self):
        return self.anchor.gl_flange_pose_list

    def setup_cc(self):
        agv = self.cc.add_cce(self.anchor.lnk_list[0])
        batt = self.cc.add_cce(self.anchor.lnk_list[1])
        batt_fx = self.cc.add_cce(self.anchor.lnk_list[2])
        self.cc.enable_extcd_by_id_list(id_list=[agv, batt, batt_fx], type="from")

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
                       name='shuidi_stickmodel'):
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
                      name='shuidi_meshmodel'):
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
    robot = Shuidi(home_pos=rm.vec(.2,.3,0), home_rotmat=rm.rotmat_from_axangle(ax=rm.const.z_ax, angle=rm.pi/12), enable_cc=True)
    # robot.goto_given_conf(conf=rm.vec(1,1,rm.pi/3))
    robot.gen_stickmodel().attach_to(base)
    robot.gen_meshmodel(alpha=.3, toggle_cdprim=True, toggle_flange_frame=True).attach_to(base)
    base.run()
