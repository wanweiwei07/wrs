import os
import math
import numpy as np
import basis.robotmath as rm
import robotsim._kinematics.jlchain as jl
from panda3d.core import NodePath, CollisionTraverser, CollisionHandlerQueue, BitMask32

class UR3E(jl.JLChain):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(6), name='ur3e'):
        super().__init__(pos=pos, rotmat=rotmat, homeconf=homeconf, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # six joints, njnts = 6+2 (tgt ranges from 1-6), nlinks = 6+1
        self.jnts[1]['loc_pos'] = np.array([0, 0, .152])
        self.jnts[2]['loc_pos'] = np.array([0, 0.12, 0])
        self.jnts[2]['loc_rotmat'] = rm.rotmat_from_euler(.0, math.pi / 2.0, .0)
        self.jnts[2]['loc_motionax'] = np.array([0, 1, 0])
        self.jnts[3]['loc_pos'] = np.array([0, -.093, .244])
        self.jnts[3]['loc_motionax'] = np.array([0, 1, 0])
        self.jnts[4]['loc_pos'] = np.array([.0, .0, 0.213])
        self.jnts[4]['loc_rotmat'] = rm.rotmat_from_euler(.0, math.pi/2.0, 0)
        self.jnts[4]['loc_motionax'] = np.array([0, 1, 0])
        self.jnts[5]['loc_pos'] = np.array([0, .104, 0])
        self.jnts[5]['loc_motionax'] = np.array([0, 0, 1])
        self.jnts[6]['loc_pos'] = np.array([0, 0, .085])
        self.jnts[6]['loc_motionax'] = np.array([0, 1, 0])
        self.jnts[7]['loc_pos'] = np.array([0, .092, 0])
        self.jnts[7]['loc_rotmat'] = rm.rotmat_from_euler(-math.pi/2.0, 0, 0)
        # links
        self.lnks[0]['name'] = "base"
        self.lnks[0]['loc_pos'] = np.zeros(3)
        self.lnks[0]['mass'] = 2.0
        self.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "base.dae")
        self.lnks[0]['rgba'] = [.5,.5,.5, 1.0]
        self.lnks[1]['name'] = "shoulder"
        self.lnks[1]['loc_pos'] = np.zeros(3)
        self.lnks[1]['com'] = np.array([.0, -.02, .0])
        self.lnks[1]['mass'] = 1.95
        self.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "shoulder.dae")
        self.lnks[1]['rgba'] = [.1,.3,.5, 1.0]
        self.lnks[2]['name'] = "upperarm"
        self.lnks[2]['loc_pos'] = np.array([.0, .0, .0])
        self.lnks[2]['com'] = np.array([.13, 0, .1157])
        self.lnks[2]['mass'] = 3.42
        self.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "upperarm.dae")
        self.lnks[2]['rgba'] = [.7,.7,.7, 1.0]
        self.lnks[3]['name'] = "forearm"
        self.lnks[3]['loc_pos'] = np.array([.0, .0, .0])
        self.lnks[3]['com'] = np.array([.05, .0, .0238])
        self.lnks[3]['mass'] = 1.437
        self.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "forearm.dae")
        self.lnks[3]['rgba'] = [.35,.35,.35, 1.0]
        self.lnks[4]['name'] = "wrist1"
        self.lnks[4]['loc_pos'] = np.array([.0, .0, .0])
        self.lnks[4]['com'] = np.array([.0, .0, 0.01])
        self.lnks[4]['mass'] = 0.871
        self.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "wrist1.dae")
        self.lnks[4]['rgba'] = [.7,.7,.7, 1.0]
        self.lnks[5]['name'] = "wrist2"
        self.lnks[5]['loc_pos'] = np.array([.0, .0, .0])
        self.lnks[5]['com'] = np.array([.0, .0, 0.01])
        self.lnks[5]['mass'] = 0.8
        self.lnks[5]['meshfile'] = os.path.join(this_dir, "meshes", "wrist2.dae")
        self.lnks[5]['rgba'] = [.1,.3,.5, 1.0]
        self.lnks[6]['name'] = "wrist3"
        self.lnks[6]['loc_pos'] = np.array([.0, .0, .0])
        self.lnks[6]['com'] = np.array([.0, .0, -0.02])
        self.lnks[6]['mass'] = 0.8
        self.lnks[6]['meshfile'] = os.path.join(this_dir, "meshes", "wrist3.dae")
        self.lnks[6]['rgba'] = [.5,.5,.5, 1.0]
        self.reinitialize()
        self.linkcdpairs = [[[0,1], [3,5,6]], [[2], [4,5,6]], [[3], [6]]]

    def is_selfcollided(self, toggledebug=False):
        for linkcdpair in self.linkcdpairs:
            oocnp = NodePath("collision nodepath")
            obj0cnp_list = []
            for one_lcdp in linkcdpair[0]:
                this_collisionmodel = self.lnks[one_lcdp]['collisionmodel']
                pos = self.lnks[one_lcdp]['gl_pos']
                rotmat = self.lnks[one_lcdp]['gl_rotmat']
                obj0cnp_list.append(this_collisionmodel.copy_cdnp_to(oocnp, rm.homomat_from_posrot(pos, rotmat)))
            obj1cnp_list = []
            for one_lcdp in linkcdpair[1]:
                this_collisionmodel = self.lnks[one_lcdp]['collisionmodel']
                pos = self.lnks[one_lcdp]['gl_pos']
                rotmat = self.lnks[one_lcdp]['gl_rotmat']
                obj1cnp_list.append(this_collisionmodel.copy_cdnp_to(oocnp, rm.homomat_from_posrot(pos, rotmat)))
            if toggledebug:
                oocnp.reparentTo(base.render)
                for obj0cnp in obj0cnp_list:
                    obj0cnp.show()
                for obj1cnp in obj1cnp_list:
                    obj1cnp.show()
            ctrav = CollisionTraverser()
            chan = CollisionHandlerQueue()
            for obj0cnp in obj0cnp_list:
                obj0cnp.node().setFromCollideMask(BitMask32(0x1))
                obj0cnp.setCollideMask(BitMask32(0x2))
                ctrav.addCollider(obj0cnp, chan)
            ctrav.traverse(oocnp)
            if chan.getNumEntries() > 0:
                if toggledebug:
                    print(linkcdpair)
                return True
        return False


if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[2, 0, 1], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    manipulator_instance = UR3E()
    manipulator_meshmodel = manipulator_instance.gen_meshmodel()
    manipulator_meshmodel.attach_to(base)
    manipulator_instance.gen_stickmodel(togglejntscs=True).attach_to(base)
    tic = time.time()
    print(manipulator_instance.is_selfcollided(toggledebug=True))
    toc = time.time()
    print(toc - tic)

    # base = wd.World(campos=[1, 1, 1], lookatpos=[0,0,0])
    # gm.GeometricModel("./meshes/base.dae").attach_to(base)
    # gm.gen_frame().attach_to(base)
    base.run()