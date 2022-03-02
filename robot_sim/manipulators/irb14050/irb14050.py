import os
import copy
import math
import numpy as np
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.manipulator_interface as mi


class IRB14050(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='irb14050', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        # seven joints, n_jnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1
        jnt_safemargin = math.pi / 18.0
        # self.jlc.jnts[1]['loc_pos'] = np.array([0.05355, -0.0725, 0.41492])
        # self.jlc.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(-0.9795, -0.5682, -2.3155)
        self.jlc.jnts[1]['loc_pos'] = np.array([0., 0., 0.])
        self.jlc.jnts[1]['motion_rng'] = [-2.94087978961 + jnt_safemargin, 2.94087978961 - jnt_safemargin]
        self.jlc.jnts[2]['loc_pos'] = np.array([0.03, 0.0, 0.1])
        self.jlc.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, 0.0, 0.0)
        self.jlc.jnts[2]['motion_rng'] = [-2.50454747661 + jnt_safemargin, 0.759218224618 - jnt_safemargin]
        self.jlc.jnts[3]['loc_pos'] = np.array([-0.03, 0.17283, 0.0])
        self.jlc.jnts[3]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jlc.jnts[3]['motion_rng'] = [-2.94087978961 + jnt_safemargin, 2.94087978961 - jnt_safemargin]
        self.jlc.jnts[4]['loc_pos'] = np.array([-0.04188, 0.0, 0.07873])
        self.jlc.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, -1.57079632679, 0.0)
        self.jlc.jnts[4]['motion_rng'] = [-2.15548162621 + jnt_safemargin, 1.3962634016 - jnt_safemargin]
        self.jlc.jnts[5]['loc_pos'] = np.array([0.0405, 0.16461, 0.0])
        self.jlc.jnts[5]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jlc.jnts[5]['motion_rng'] = [-5.06145483078 + jnt_safemargin, 5.06145483078 - jnt_safemargin]
        self.jlc.jnts[6]['loc_pos'] = np.array([-0.027, 0, 0.10039])
        self.jlc.jnts[6]['loc_rotmat'] = rm.rotmat_from_euler(1.57079632679, 0.0, 0.0)
        self.jlc.jnts[6]['motion_rng'] = [-1.53588974176 + jnt_safemargin, 2.40855436775 - jnt_safemargin]
        self.jlc.jnts[7]['loc_pos'] = np.array([0.027, 0.029, 0.0])
        self.jlc.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-1.57079632679, 0.0, 0.0)
        self.jlc.jnts[7]['motion_rng'] = [-3.99680398707 + jnt_safemargin, 3.99680398707 - jnt_safemargin]
        # links
        self.jlc.lnks[1]['name'] = "link_1"
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link_1.stl")
        self.jlc.lnks[1]['rgba'] = [.5, .5, .5, 1]
        self.jlc.lnks[2]['name'] = "link_2"
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link_2.stl")
        self.jlc.lnks[2]['rgba'] = [.929, .584, .067, 1]
        self.jlc.lnks[3]['name'] = "link_3"
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link_3.stl")
        self.jlc.lnks[3]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[4]['name'] = "link_4"
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link_4.stl")
        self.jlc.lnks[4]['rgba'] = [0.180, .4, 0.298, 1]
        self.jlc.lnks[5]['name'] = "link_5"
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link_5.stl")
        self.jlc.lnks[5]['rgba'] = [.7, .7, .7, 1]
        self.jlc.lnks[6]['name'] = "link_6"
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link_6.stl")
        self.jlc.lnks[6]['rgba'] = [0.180, .4, 0.298, 1]
        self.jlc.lnks[7]['name'] = "link_7"
        # self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "link_7.stl") # not really needed to visualize
        # self.jlc.lnks[7]['rgba'] = [.5,.5,.5,1]
        # reinitialization
        self.jlc.reinitialize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [1, 2, 3, 4, 5, 6])
        activelist = [self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[1]]
        intolist = [self.jlc.lnks[5],
                    self.jlc.lnks[6]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import copy
    import time
    import copy
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 0, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = IRB14050(enable_cc=True)
    manipulator_instance.fk(
        jnt_values=[0, 0, manipulator_instance.jnts[3]['motion_rng'][1] / 2, manipulator_instance.jnts[4]['motion_rng'][1], 0, 0, 0])
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel().attach_to(base)
    manipulator_instance.show_cdprimit()
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)

    manipulator_instance2 = manipulator_instance.copy()
    # manipulator_instance2.disable_localcc()
    manipulator_instance2.fix_to(pos=np.array([.2, .2, 0.2]), rotmat=np.eye(3))
    manipulator_instance2.fk(
        jnt_values=[0, 0, manipulator_instance.jnts[3]['motion_rng'][1] / 2, manipulator_instance.jnts[4]['motion_rng'][1]*1.1,
                    manipulator_instance.jnts[5]['motion_rng'][1], 0, 0])
    manipulator_meshmodel2 = manipulator_instance2.gen_meshmodel()
    manipulator_meshmodel2.attach_to(base)
    manipulator_instance2.show_cdprimit()
    tic = time.time()
    print(manipulator_instance2.is_collided())
    toc = time.time()
    print(toc - tic)
    base.run()
