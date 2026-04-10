import os
import math
import numpy as np
import wrs.basis.robot_math as rm
import wrs.modeling.geometric_model as mgm
import wrs.modeling.collision_model as mcm
import wrs.robot_sim.manipulators.manipulator_interface as mi


class UR5E(mi.ManipulatorInterface):
    """
    UR5e manipulator using the modern WRS JLChain API.

    DH parameters from Universal Robots UR5e datasheet.
    Mesh files: .dae format in the meshes/ directory.
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), home_conf=np.zeros(6),
                 name='ur5e', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor (base link)
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "base.dae"), name="ur5e_base")
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.5, .5, .5, 1.0])
        # Joint 0 (base rotation) and link
        self.jlc.jnts[0].loc_pos = np.array([0, 0, 0.163])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "shoulder.dae"), name="ur5e_shoulder")
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # Joint 1 (shoulder) and link
        self.jlc.jnts[1].loc_pos = np.array([0, 0.138, 0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(0, math.pi / 2.0, 0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "upperarm.dae"), name="ur5e_upperarm")
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # Joint 2 (elbow) and link
        self.jlc.jnts[2].loc_pos = np.array([0, -0.131, 0.425])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "forearm.dae"), name="ur5e_forearm")
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.35, .35, .35, 1.0])
        # Joint 3 (wrist 1) and link
        self.jlc.jnts[3].loc_pos = np.array([0, 0, 0.392])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(0, math.pi / 2.0, 0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[3].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wrist1.dae"), name="ur5e_wrist1")
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # Joint 4 (wrist 2) and link
        self.jlc.jnts[4].loc_pos = np.array([0, 0.127, 0])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wrist2.dae"), name="ur5e_wrist2")
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # Joint 5 (wrist 3) and link
        self.jlc.jnts[5].loc_pos = np.array([0, 0, 0.100])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[5].motion_range = np.array([-2 * math.pi, 2 * math.pi])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(
            initor=os.path.join(current_file_dir, "meshes", "wrist3.dae"), name="ur5e_wrist3")
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.5, .5, .5, 1.0])
        # flange (TCP frame offset)
        self.jlc._loc_flange_pos = np.array([0, 0.100, 0])
        self.jlc._loc_flange_rotmat = rm.rotmat_from_euler(-math.pi / 2.0, 0, 0)
        # finalize with numeric IK solver
        self.jlc.finalize(ik_solver='n', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, 0, 0])
        self.loc_tcp_rotmat = np.eye(3)
        # set up collision checking
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        """Set up collision checking pairs (following ur3e pattern)."""
        lb = self.cc.add_cce(self.jlc.anchor.lnk_list[0])
        l0 = self.cc.add_cce(self.jlc.jnts[0].lnk)
        l1 = self.cc.add_cce(self.jlc.jnts[1].lnk)
        l2 = self.cc.add_cce(self.jlc.jnts[2].lnk)
        l3 = self.cc.add_cce(self.jlc.jnts[3].lnk)
        l4 = self.cc.add_cce(self.jlc.jnts[4].lnk)
        l5 = self.cc.add_cce(self.jlc.jnts[5].lnk)
        # Distal links vs proximal links (skip adjacent)
        from_list = [l3, l4, l5]
        into_list = [lb, l0]
        self.cc.set_cdpair_by_ids(from_list, into_list)


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    mgm.gen_frame().attach_to(base)
    manipulator_instance = UR5E(enable_cc=True)
    manipulator_instance.gen_meshmodel(alpha=.3).attach_to(base)
    manipulator_instance.gen_stickmodel(toggle_jnt_frames=True).attach_to(base)
    print(f"n_dof: {manipulator_instance.n_dof}")
    print(f"jnt_ranges: {manipulator_instance.jnt_ranges}")
    print(f"is_collided: {manipulator_instance.is_collided()}")
    base.run()