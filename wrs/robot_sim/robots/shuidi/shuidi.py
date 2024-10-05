import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.robot_sim.robots.robot_interface as ri
import wrs.robot_sim._kinematics.jlchain as rkjlc
import wrs.modeling.collision_model as mcm
import wrs.modeling.model_collection as mmc

class Shuidi(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="XYInterface", enable_cc=False):
        """
        2D collision detection when enable_cc is False
        :param pos:
        :param rotmat:
        :param name:
        :param enable_cc:
        """
        super().__init__(pos=pos, rotmat=rotmat, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # jlc
        self.jlc = rkjlc.JLChain(n_dof=3, name=name)
        self.jlc.home = np.zeros(3)
        self.jlc.jnts[0].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.jlc.jnts[0].loc_motion_ax = np.array([1, 0, 0])
        self.jlc.jnts[0].loc_pos = np.zeros(3)
        self.jlc.jnts[0].motion_range = np.array([-15.0, 15.0])
        self.jlc.jnts[1].change_type(type=rkjlc.const.JntType.PRISMATIC)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].loc_pos = np.zeros(3)
        self.jlc.jnts[1].motion_range = np.array([-15.0, 15.0])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[2].loc_pos = np.zeros(3)
        self.jlc.jnts[2].motion_range = [-math.pi, math.pi]
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "shuidi_agv.stl"), cdprim_type=mcm.const.CDPrimType.CYLINDER)
        self.jlc.jnts[2].lnk.cmodel.rgba = rm.const.tab20_list[14]
        self.jlc.finalize()
        # anchor
        self.anchor = rkjlc.rkjl.Anchor(name=name + "_anchor", pos=self.jlc.gl_flange_pos,
                                        rotmat=self.jlc.gl_flange_rotmat, n_flange=1, n_lnk=2)
        # anchor flange
        self.anchor.loc_flange_pose_list[0] = [np.array([.0, .0, .445862]), np.eye(3)]
        # anchor battery
        self.anchor.lnk_list[0].name = name + "_battery"
        self.anchor.lnk_list[0].loc_pos = np.array([.0, .0, .277])
        self.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "battery.stl"))
        self.anchor.lnk_list[0].cmodel.rgba = rm.const.tab20_list[14]
        # anchor battery fixture
        self.anchor.lnk_list[1].name = name + "_battery_fixture"
        self.anchor.lnk_list[1].loc_pos = np.array([.0, .0, .277])
        self.anchor.lnk_list[1].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "battery_fixture.stl"))
        self.anchor.lnk_list[1].cmodel.rgba = rm.const.tab20_list[15]
        if enable_cc:
            self.setup_cc()
        # backup
        self.jnt_values_bk = []

    @property
    def n_dof(self):
        return self.jlc.n_dof

    @property
    def gl_flange_pose_list(self):
        return self.anchor.gl_flange_pose_list

    def setup_cc(self):
        self.cc.add_cce(self.jlc.jnts[2].lnk)
        self.cc.add_cce(self.anchor.lnk_list[0])
        self.cc.add_cce(self.anchor.lnk_list[1])

    def backup_state(self):
        self.jnt_values_bk.append(self.jlc.get_jnt_values())

    def restore_state(self):
        self.jlc.goto_given_conf(jnt_values=self.jnt_values_bk.pop())
        self.anchor.pos = self.jlc.gl_flange_pos
        self.anchor.rotmat = self.jlc.gl_flange_rotmat

    def goto_given_conf(self, jnt_values=np.zeros(2)):
        self.jlc.goto_given_conf(jnt_values=jnt_values)
        self.anchor.pos = self.jlc.gl_flange_pos
        self.anchor.rotmat = self.jlc.gl_flange_rotmat

    def rand_conf(self):
        return self.jlc.rand_conf()

    def get_jnt_values(self):
        return self.jlc.get_jnt_values()

    def are_jnts_in_ranges(self, jnt_values):
        return self.jlc.are_jnts_in_ranges(jnt_values)

    def is_collided(self, obstacle_list=None, other_robot_list=None, toggle_contacts=False):
        collision_info = self.cc.is_collided(obstacle_list=obstacle_list, other_robot_list=other_robot_list,
                                             toggle_contacts=toggle_contacts)
        return collision_info

    def gen_stickmodel(self,
                       toggle_tcp_frame=False,
                       toggle_jnt_frames=False,
                       toggle_flange_frame=False,
                       name='shuidi_stickmodel'):
        m_col = mmc.ModelCollection(name=name)
        self.jlc.gen_stickmodel(toggle_jnt_frames=toggle_jnt_frames,
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
        self.jlc.gen_meshmodel(rgb=rgb, alpha=alpha, toggle_flange_frame=False,
                               toggle_jnt_frames=toggle_jnt_frames,
                               toggle_cdmesh=toggle_cdmesh,
                               toggle_cdprim=toggle_cdprim).attach_to(m_col)
        self.anchor.gen_meshmodel(name + "_anchor", rgb=rgb,
                                  alpha=alpha, toggle_cdmesh=toggle_cdmesh, toggle_cdprim=toggle_cdprim,
                                  toggle_root_frame=toggle_jnt_frames,
                                  toggle_flange_frame=toggle_flange_frame).attach_to(m_col)
        return m_col


if __name__ == "__main__":
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])
    mcm.mgm.gen_frame().attach_to(base)
    robot = Shuidi(enable_cc=True)
    robot.goto_given_conf(jnt_values=np.array([1, 1, math.pi / 3]))
    robot.gen_stickmodel().attach_to(base)
    robot.gen_meshmodel(alpha=.3, toggle_cdprim=True, toggle_flange_frame=True).attach_to(base)
    base.run()
