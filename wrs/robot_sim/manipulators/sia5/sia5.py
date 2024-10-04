import os
import math
import numpy as np
from wrs import basis as rm, robot_sim as jl, robot_sim as mi, modeling as gm


class SIA5(mi.ManipulatorInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='sia5', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        self.jlc = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=homeconf, name=name)
        # seven joints, n_jnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1
        jnt_safemargin = math.pi / 18.0
        self.jlc.jnts[1]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.jnts[1]['motion_range'] = [-math.pi + jnt_safemargin, math.pi]
        self.jlc.jnts[2]['loc_pos'] = np.array([0, 0, .168])
        self.jlc.jnts[2]['motion_range'] = [-math.radians(110) + jnt_safemargin, math.radians(110) - jnt_safemargin]
        self.jlc.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jlc.jnts[3]['loc_pos'] = np.array([0, 0, .27])
        self.jlc.jnts[3]['motion_range'] = [-math.radians(170) + jnt_safemargin, math.radians(170) - jnt_safemargin]
        self.jlc.jnts[4]['loc_pos'] = np.array([.085, 0, 0])
        self.jlc.jnts[4]['motion_range'] = [-math.radians(90) + jnt_safemargin, math.radians(115) - jnt_safemargin]
        self.jlc.jnts[4]['loc_motionax'] = np.array([0, -1, 0])
        self.jlc.jnts[5]['loc_pos'] = np.array([.27, -0, .06])
        self.jlc.jnts[5]['motion_range'] = [-math.radians(90) + jnt_safemargin, math.radians(90) - jnt_safemargin]
        self.jlc.jnts[5]['loc_motionax'] = np.array([-1, 0, 0])
        self.jlc.jnts[6]['loc_pos'] = np.array([0, 0, 0])
        self.jlc.jnts[6]['motion_range'] = [-math.radians(110) + jnt_safemargin, math.radians(110) - jnt_safemargin]
        self.jlc.jnts[6]['loc_motionax'] = np.array([0, -1, 0])
        self.jlc.jnts[7]['loc_pos'] = np.array([.134, .0, .0])
        self.jlc.jnts[7]['gl_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, 0, 0)
        self.jlc.jnts[7]['motion_range'] = [-math.radians(90) + jnt_safemargin, math.radians(90) - jnt_safemargin]
        self.jlc.jnts[7]['loc_motionax'] = np.array([-1, 0, 0])
        self.jlc.jnts[8]['loc_pos'] = np.array([.011, .0, .0])
        self.jlc.jnts[8]['gl_rotmat'] = rm.rotmat_from_euler(-math.pi / 2, -math.pi / 2, -math.pi / 2)
        # links
        # self.jlc.lnks[0]['name'] = "base"
        # self.jlc.lnks[0]['loc_pos'] = np.zeros(3)
        # self.jlc.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "base.dae")
        # self.jlc.lnks[0]['rgba'] = [.5,.5,.5, 1.0]
        self.jlc.lnks[1]['name'] = "link_s"
        self.jlc.lnks[1]['loc_pos'] = np.array([0, 0, .142])
        self.jlc.lnks[1]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, "meshes", "link_s.dae")
        self.jlc.lnks[1]['rgba'] = [.55,.55,.55, 1.0]
        self.jlc.lnks[2]['name'] = "link_l"
        self.jlc.lnks[2]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[2]['gl_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.jlc.lnks[2]['mesh_file'] = os.path.join(this_dir, "meshes", "link_l.dae")
        self.jlc.lnks[2]['rgba'] = [.1,.3,.5, 1.0]
        self.jlc.lnks[3]['name'] = "link_e"
        self.jlc.lnks[3]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[3]['mesh_file'] = os.path.join(this_dir, "meshes", "link_e.dae")
        self.jlc.lnks[3]['rgba'] = [.7,.7,.7, 1.0]
        self.jlc.lnks[4]['name'] = "link_u"
        self.jlc.lnks[4]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[4]['mesh_file'] = os.path.join(this_dir, "meshes", "link_u.dae")
        self.jlc.lnks[4]['rgba'] = [.1,.3,.5, 1.0]
        self.jlc.lnks[5]['name'] = "link_r"
        self.jlc.lnks[5]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[5]['mesh_file'] = os.path.join(this_dir, "meshes", "link_r.dae")
        self.jlc.lnks[5]['rgba'] = [.7,.7,.7, 1.0]
        self.jlc.lnks[6]['name'] = "link_b"
        self.jlc.lnks[6]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[6]['mesh_file'] = os.path.join(this_dir, "meshes", "link_b.dae")
        self.jlc.lnks[6]['rgba'] = [.1,.3,.5, 1.0]
        self.jlc.lnks[7]['name'] = "link_t"
        self.jlc.lnks[7]['loc_pos'] = np.zeros(3)
        self.jlc.lnks[7]['mesh_file'] = os.path.join(this_dir, "meshes", "link_t.dae")
        self.jlc.lnks[7]['rgba'] = [.7,.7,.7, 1.0]
        # reinitialization
        # self.jlc.setinitvalues(np.array([-math.pi/2, math.pi/3, math.pi/6, 0, 0, 0, 0]))
        # self.jlc.setinitvalues(np.array([-math.pi/2, 0, math.pi/3, math.pi/10, 0, 0, 0]))
        self.jlc.finalize()
        # collision detection
        if enable_cc:
            self.enable_cc()

    def enable_cc(self):
        super().enable_cc()
        self.cc.add_cdlnks(self.jlc, [1, 2, 3, 4, 5, 6, 7])
        activelist = [self.jlc.lnks[1],
                      self.jlc.lnks[2],
                      self.jlc.lnks[3],
                      self.jlc.lnks[4],
                      self.jlc.lnks[5],
                      self.jlc.lnks[6],
                      self.jlc.lnks[7]]
        self.cc.set_active_cdlnks(activelist)
        fromlist = [self.jlc.lnks[1],
                    self.jlc.lnks[2]]
        intolist = [self.jlc.lnks[5],
                    self.jlc.lnks[6],
                    self.jlc.lnks[7]]
        self.cc.set_cdpair(fromlist, intolist)


if __name__ == '__main__':
    import time
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[2, 0, 1], lookat_pos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    manipulator_instance = SIA5(enable_cc=True)
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_meshmodel.show_cdprim()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    tic = time.time()
    print(manipulator_instance.is_collided())
    toc = time.time()
    print(toc - tic)
    base.run()
