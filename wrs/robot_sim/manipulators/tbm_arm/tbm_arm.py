import os
import math
import numpy as np
from wrs import robot_sim as jl, robot_sim as mi, modeling as gm


class TBMArm(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='tbm_arm', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)
        # six joints, n_jnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        # self.jlc.joints[1]['loc_pos'] = np.array([0.2, 0, 0])
        # self.jlc.joints[1]['end_type'] = 'prismatic'
        # self.jlc.joints[1]['loc_motionax'] = np.array([1, 0, 0])
        # self.jlc.joints[1]['motion_range'] = [-.1, 1.25]
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0.346])
        self.jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[1]['motion_range'] = [-math.radians(60), math.radians(60)]
        self.jlc.jnts[2]['loc_pos'] = np.array([0.645, .0, .0])
        self.jlc.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        # self.jlc.joints[2]['motion_range'] = [-2*math.pi, 2*math.pi]
        self.jlc.jnts[3]['loc_pos'] = np.array([.425, .0, .0])
        self.jlc.jnts[3]['loc_motionax'] = np.array([0, 0, 1])
        self.jlc.jnts[3]['motion_range'] = [-math.radians(90), math.radians(90)]
        self.jlc.jnts[4]['loc_pos'] = np.array([0.587, .0, .0])
        self.jlc.jnts[4]['loc_motionax'] = np.array([1, 0, 0])
        # self.jlc.joints[4]['motion_range'] = [-2*math.pi, 2*math.pi]
        self.jlc.jnts[5]['loc_pos'] = np.array([.63, .0, .0])
        self.jlc.jnts[5]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[5]['motion_range'] = [-math.radians(115), math.radians(115)]
        self.jlc.jnts[6]['loc_pos'] = np.array([.329, .0, .0])
        self.jlc.jnts[6]['loc_motionax'] = np.array([1, 0, 0])
        # self.jlc.joints[6]['motion_range'] = [-2*math.pi, 2*math.pi]
        # links
        self.jlc.lnks[0]['name'] = "base"
        self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        # self.jlc.lnks[0]['mass'] = 1.4
        # self.jlc.lnks[0]['com'] = np.array([-.02131, .000002, .044011])
        self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.stl")
        self.jlc.lnks[0]['rgba'] = [.5, .5, .5, 1.0]
        self.jlc.lnks[1]['name'] = "j1"
        self.jlc.lnks[1]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[1]['com'] = np.array([.0, .0, .15])
        self.jlc.lnks[1]['mass'] = 1.29
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "joint1.stl")
        self.jlc.lnks[1]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.lnks[2]['name'] = "j2"
        self.jlc.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[2]['com'] = np.array([-.02, .1, .07])
        self.jlc.lnks[2]['mass'] = 0.39
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "joint2.stl")
        self.jlc.lnks[2]['rgba'] = [.77, .77, .60, 1]
        self.jlc.lnks[3]['name'] = "j3"
        self.jlc.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[3]['com'] = np.array([-.01, .02, .03])
        self.jlc.lnks[3]['mass'] = .35
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "joint3.stl")
        self.jlc.lnks[3]['rgba'] = [.35, .35, .35, 1.0]
        self.jlc.lnks[4]['name'] = "j4"
        self.jlc.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[4]['com'] = np.array([.0, .0, 0.055])
        self.jlc.lnks[4]['mass'] = 0.35
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "joint4.stl")
        self.jlc.lnks[4]['rgba'] = [.7, .7, .7, 1.0]
        self.jlc.lnks[5]['name'] = "j5"
        self.jlc.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[5]['com'] = np.array([.0, -.04, .015])
        self.jlc.lnks[5]['mass'] = 0.19
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "joint5.stl")
        self.jlc.lnks[5]['rgba'] = [.77, .77, .60, 1]
        self.jlc.lnks[6]['name'] = "j6"
        self.jlc.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.jlc.lnks[6]['com'] = np.array([.0, .0, 0])
        self.jlc.lnks[6]['mass'] = 0.03
        self.jlc.lnks[6]['mesh_file'] = None
        self.jlc.lnks[6]['rgba'] = [.5, .5, .5, 1.0]
        self.jlc.finalize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        # self.cc.add_cdlnks(self.jlc, [0, 1, 2, 3, 4, 5, 6, 7])
        # activelist = [self.jlc.lnks[0],
        #               self.jlc.lnks[1],
        #               self.jlc.lnks[2],
        #               self.jlc.lnks[3],
        #               self.jlc.lnks[4],
        #               self.jlc.lnks[5],
        #               self.jlc.lnks[6]]
        # self.cc.set_active_cdlnks(activelist)
        # fromlist = [self.jlc.lnks[0],
        #             self.jlc.lnks[1]]
        # into_list = [self.jlc.lnks[3],
        #             self.jlc.lnks[5],
        #             self.jlc.lnks[6]]
        # self.cc.set_cdpair(fromlist, into_list)
        # fromlist = [self.jlc.lnks[2]]
        # into_list = [self.jlc.lnks[4],
        #             self.jlc.lnks[5],
        #             self.jlc.lnks[6]]
        # self.cc.set_cdpair(fromlist, into_list)
        # fromlist = [self.jlc.lnks[3]]
        # into_list = [self.jlc.lnks[6]]
        # self.cc.set_cdpair(fromlist, into_list)


if __name__ == '__main__':
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[-5, -5, 3], lookat_pos=[3, 0, 0])
    gm.gen_frame().attach_to(base)
    seed0 = np.zeros(6)
    seed0[3] = math.pi / 2
    seed1 = np.zeros(6)
    seed1[3] = -math.pi / 2
    manipulator_instance = TBMArm(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel(toggle_jnt_frames=True)
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    # base.run()
    seed_jnt_values = manipulator_instance.get_jnt_values()
    for x in np.linspace(1, 3, 7).tolist():
        for y in np.linspace(-2, 2, 14).tolist():
            print(x)
            tgt_pos = np.array([x, y, 0])
            tgt_rotmat = np.eye(3)
            jnt_values0 = manipulator_instance.ik(tgt_pos, tgt_rotmat, max_niter=100, toggle_dbg=False,
                                                  seed_jnt_values=seed0)
            if jnt_values0 is not None:
                jnt_values = jnt_values0
            else:
                jnt_values1 = manipulator_instance.ik(tgt_pos, tgt_rotmat, max_niter=100, toggle_dbg=False,
                                                      seed_jnt_values=seed1)
                if jnt_values1 is not None:
                    jnt_values = jnt_values1
                else:
                    jnt_values = None
            if jnt_values is not None:
                # last_jnt_values = jnt_values
                gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat, axis_radius=.02).attach_to(base)
                manipulator_instance.fk(jnt_values=jnt_values)
                manipulator_instance.gen_meshmodel().attach_to(base)
            else:
                gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    base.run()

    print(jnt_values)
    manipulator_instance.fk(jnt_values=jnt_values)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_meshmodel.show_cdprim()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    # tic = time.time()
    # print(arm.is_collided())
    # toc = time.time()
    # print(toc - tic)

    # base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0,0,0])
    # mgm.GeometricModel("./meshes/base.dae").attach_to(base)
    # mgm.gen_frame().attach_to(base)
    base.run()
