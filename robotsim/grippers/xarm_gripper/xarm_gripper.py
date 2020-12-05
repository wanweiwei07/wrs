import numpy as np
import robotsim._kinematics.jlchain as jl
import modeling.geometricmodel as gm

class XArmGripper(object):

    def __init__(self):
        self.lft_outer = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(2), name='lft_outer')
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, .035, .059098])
        self.lft_outer.jnts[1]['rngmin'] = .0
        self.lft_outer.jnts[1]['rngmax'] = .85 # TODO change min-max to a tuple
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, .035465, .042039]) # passive
        self.lft_outer.jnts[2]['rngmin'] = .0
        self.lft_outer.jnts[2]['rngmax'] = .85 # TODO change min-max to a tuple
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft_outer.reinitialize()
        self.lft_inner = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(1), name='lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, .02, .074098])
        self.lft_inner.jnts[1]['rngmin'] = .0
        self.lft_inner.jnts[1]['rngmax'] = .85 # TODO change min-max to a tuple
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_inner.reinitialize()
        self.rgt_outer = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(2), name='rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, -.035, .059098])
        self.rgt_outer.jnts[1]['rngmin'] = .0
        self.rgt_outer.jnts[1]['rngmax'] = .85 # TODO change min-max to a tuple
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, -.035465, .042039]) # passive
        self.rgt_outer.jnts[2]['rngmin'] = .0
        self.rgt_outer.jnts[2]['rngmax'] = .85 # TODO change min-max to a tuple
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.reinitialize()
        self.rgt_inner = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(1), name='rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, -.02, .074098])
        self.rgt_inner.jnts[1]['rngmin'] = .0
        self.rgt_inner.jnts[1]['rngmax'] = .85 # TODO change min-max to a tuple
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt_inner.reinitialize()

    def gen_stickmodel(self, name='xarm_gripper_stickmodel'):
        stickmodel = gm.StaticGeometricModel(name=name)
        self.lft_outer.gen_stickmodel().attach_to(stickmodel)
        self.lft_inner.gen_stickmodel().attach_to(stickmodel)
        self.rgt_outer.gen_stickmodel().attach_to(stickmodel)
        self.rgt_inner.gen_stickmodel().attach_to(stickmodel)
        return stickmodel

if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(camp=[2, 0, 1], lookatpos=[0, 0, 0.5])
    gm.gen_frame().attach_to(base)
    xag = XArmGripper()
    xag.gen_stickmodel().attach_to(base)
    base.run()