import os
import math
import numpy as np
import modeling.collision_model as mcm
import basis.robot_math as rm
import robot_sim.manipulators.manipulator_interface as mi


class UR3(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), home_conf=np.zeros(6), name='ur3', enable_cc=False):
        super().__init__(pos=pos, rotmat=rotmat, home_conf=home_conf, name=name, enable_cc=enable_cc)
        current_file_dir = os.path.dirname(__file__)
        # anchor
        self.jlc.anchor.lnk_list[0].cmodel = mcm.CollisionModel(
            os.path.join(current_file_dir, "meshes", "base.stl"))
        self.jlc.anchor.lnk_list[0].cmodel.rgba = np.array([.5, .5, .5, 1.0])
        # first joint and link
        self.jlc.jnts[0].loc_pos = np.array([.0, .0, .1519])
        self.jlc.jnts[0].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[0].motion_range = np.array([-math.pi * 2, math.pi * 2])
        self.jlc.jnts[0].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "shoulder.stl"))
        self.jlc.jnts[0].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # second joint and link
        self.jlc.jnts[1].loc_pos = np.array([.0, .1198, .0])
        self.jlc.jnts[1].loc_rotmat = rm.rotmat_from_euler(.0, math.pi / 2.0, .0)
        self.jlc.jnts[1].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[1].motion_range = np.array([-math.pi * 2, math.pi * 2])
        self.jlc.jnts[1].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "upperarm.stl"))
        self.jlc.jnts[1].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # third joint and link
        self.jlc.jnts[2].loc_pos = np.array([.0, -.0925, .24365])
        self.jlc.jnts[2].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[2].motion_range = np.array([-math.pi, math.pi])
        self.jlc.jnts[2].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "forearm.stl"))
        self.jlc.jnts[2].lnk.cmodel.rgba = np.array([.35, .35, .35, 1.0])
        # fourth joint and link
        self.jlc.jnts[3].loc_pos = np.array([.0, .0, 0.21325])
        self.jlc.jnts[3].loc_rotmat = rm.rotmat_from_euler(.0, math.pi / 2.0, .0)
        self.jlc.jnts[3].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[3].motion_range = np.array([-math.pi * 2, math.pi * 2])
        self.jlc.jnts[3].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist1.stl"))
        self.jlc.jnts[3].lnk.cmodel.rgba = np.array([.7, .7, .7, 1.0])
        # fifth joint and link
        self.jlc.jnts[4].loc_pos = np.array([.0, 0.08505, .0])
        self.jlc.jnts[4].loc_motion_ax = np.array([0, 0, 1])
        self.jlc.jnts[4].motion_range = np.array([-math.pi * 2, math.pi * 2])
        self.jlc.jnts[4].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist2.stl"))
        self.jlc.jnts[4].lnk.cmodel.rgba = np.array([.1, .3, .5, 1.0])
        # sixth joint and link
        self.jlc.jnts[5].loc_pos = np.array([.0, .0, .08535])
        self.jlc.jnts[5].loc_motion_ax = np.array([0, 1, 0])
        self.jlc.jnts[5].motion_range = np.array([-math.pi * 2, math.pi * 2])
        self.jlc.jnts[5].lnk.cmodel = mcm.CollisionModel(os.path.join(current_file_dir, "meshes", "wrist3.stl"))
        self.jlc.jnts[5].lnk.cmodel.rgba = np.array([.5, .5, .5, 1.0])
        self.jlc.finalize(ik_solver='d', identifier_str=name)
        # tcp
        self.loc_tcp_pos = np.array([0, .0819, 0])
        self.loc_tcp_rotmat = rm.rotmat_from_euler(-math.pi / 2.0, math.pi / 2.0, 0)
        # set up cc
        if self.cc is not None:
            self.setup_cc()

    def setup_cc(self):
        pass


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    arm = UR3(enable_cc=True)
    arm.gen_meshmodel().attach_to(base)
    base.run()
