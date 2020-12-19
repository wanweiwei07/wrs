import os
import math
import numpy as np
import basis.robotmath as rm
import robotsim._kinematics.jlchain as jl
import robotsim._kinematics.collisionchecker as cc


class SIA5F(jl.JLChain):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(7), name='sia5f'):
        super().__init__(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # seven joints, njnts = 7+2 (tgt ranges from 1-7), nlinks = 7+1
        self.jnts[1]['loc_pos'] = np.array([0, 0, .31])
        self.jnts[1]['rngmin'] = -math.pi
        self.jnts[1]['rngmax'] = math.pi
        self.jnts[2]['loc_pos'] = np.array([0, 0, 0])
        self.jnts[2]['rngmin'] = -math.radians(110)
        self.jnts[2]['rngmax'] = math.radians(110)
        self.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jnts[3]['loc_pos'] = np.array([0, 0, .27])
        self.jnts[3]['rngmin'] = -math.radians(170)
        self.jnts[3]['rngmax'] = math.radians(170)
        self.jnts[4]['loc_pos'] = np.array([.085, 0, 0])
        self.jnts[4]['rngmin'] = -math.radians(90)
        self.jnts[4]['rngmax'] = math.radians(115)
        self.jnts[4]['loc_motionax'] = np.array([0, -1, 0])
        self.jnts[5]['loc_pos'] = np.array([.27, -0, .06])
        self.jnts[5]['rngmin'] = -math.radians(90)
        self.jnts[5]['rngmax'] = math.radians(90)
        self.jnts[5]['loc_motionax'] = np.array([-1, 0, 0])
        self.jnts[6]['loc_pos'] = np.array([0, 0, 0])
        self.jnts[6]['rngmin'] = -math.radians(110)
        self.jnts[6]['rngmax'] = math.radians(110)
        self.jnts[6]['loc_motionax'] = np.array([0, -1, 0])
        self.jnts[7]['loc_pos'] = np.array([.134, .0, .0])
        self.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-1.5708, 0, 0)
        self.jnts[7]['rngmin'] = -math.radians(90)
        self.jnts[7]['rngmax'] = math.radians(90)
        self.jnts[7]['loc_motionax'] = np.array([-1, 0, 0])
        self.jnts[8]['loc_pos'] = np.array([.011, .0, .0])
        self.jnts[8]['loc_rotmat'] = rm.rotmat_from_euler(-1.5708, -1.5708, -1.5708)
        # links
        self.lnks[0]['name'] = "base"
        self.lnks[0]['loc_pos'] = np.zeros(3)
        self.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "base.dae")
        self.lnks[0]['rgba'] = [.5,.5,.5, 1.0]
        self.lnks[1]['name'] = "link_s"
        self.lnks[1]['loc_pos'] = np.zeros(3)
        self.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "link_s.dae")
        self.lnks[1]['rgba'] = [.35,.35,.35, 1.0]
        self.lnks[2]['name'] = "link_l"
        self.lnks[2]['loc_pos'] = np.zeros(3)
        self.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "link_l.dae")
        self.lnks[2]['rgba'] = [.1,.3,.5, 1.0]
        self.lnks[3]['name'] = "link_e"
        self.lnks[3]['loc_pos'] = np.zeros(3)
        self.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "link_e.dae")
        self.lnks[3]['rgba'] = [.35,.35,.35, 1.0]
        self.lnks[4]['name'] = "link_u"
        self.lnks[4]['loc_pos'] = np.zeros(3)
        self.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "link_u.dae")
        self.lnks[4]['rgba'] = [.1,.3,.5, 1.0]
        self.lnks[5]['name'] = "link_r"
        self.lnks[5]['loc_pos'] = np.zeros(3)
        self.lnks[5]['meshfile'] = os.path.join(this_dir, "meshes", "link_r.dae")
        self.lnks[5]['rgba'] = [.7,.7,.7, 1.0]
        self.lnks[6]['name'] = "link_b"
        self.lnks[6]['loc_pos'] = np.zeros(3)
        self.lnks[6]['meshfile'] = os.path.join(this_dir, "meshes", "link_b.dae")
        self.lnks[6]['rgba'] = [.5,.5,.5, 1.0]
        self.lnks[7]['name'] = "link_t"
        self.lnks[7]['loc_pos'] = np.zeros(3)
        self.lnks[7]['meshfile'] = os.path.join(this_dir, "meshes", "link_t.dae")
        self.lnks[7]['rgba'] = [.7,.7,.7, 1.0]
        # reinitialization
        # self.setinitvalues(np.array([-math.pi/2, math.pi/3, math.pi/6, 0, 0, 0, 0]))
        # self.setinitvalues(np.array([-math.pi/2, 0, math.pi/3, math.pi/10, 0, 0, 0]))
        self.reinitialize()
        # collision detection
        self._mt.add_cdpair([0, 1, 2], [5, 6, 7])


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[2, 0, 1], lookatpos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    manipulator_instance = SIA5F()
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_meshmodel.show_cdprimit()
    manipulator_instance.gen_stickmodel(toggle_jntscs=True).attach_to(base)
    tic = time.time()
    print(manipulator_instance.is_selfcollided())
    toc = time.time()
    print(toc - tic)
    base.run()
