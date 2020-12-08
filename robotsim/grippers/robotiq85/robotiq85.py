import os
import math
import numpy as np
import robotsim._kinematics.jlchain as jl
import basis.robotmath as rm


class Robotiq85(object):

    def __init__(self):
        this_dir, this_filename = os.path.split(__file__)
        # joints
        # - lft_outer
        self.lft_outer = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(4), name='lft_outer')
        self.lft_outer.jnts[1]['loc_pos'] = np.array([0, -.0306011, .054904])
        self.lft_outer.jnts[1]['rngmin'] = .0
        self.lft_outer.jnts[1]['rngmax'] = .8  # TODO change min-max to a tuple
        self.lft_outer.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        self.lft_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        self.lft_outer.jnts[3]['rngmin'] = -.8757
        self.lft_outer.jnts[3]['rngmax'] = .0  # TODO change min-max to a tuple
        self.lft_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.lft_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # self.lft_outer.jnts[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # - lft_inner
        self.lft_inner = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(1), name='lft_inner')
        self.lft_inner.jnts[1]['loc_pos'] = np.array([0, -.0127, .06142])
        self.lft_inner.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft_inner.jnts[1]['rngmin'] = .0
        self.lft_inner.jnts[1]['rngmax'] = .8757  # TODO change min-max to a tuple
        self.lft_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # - rgt_outer
        self.rgt_outer = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(4), name='rgt_outer')
        self.rgt_outer.jnts[1]['loc_pos'] = np.array([0, .0306011, .054904])
        self.rgt_outer.jnts[1]['rngmin'] = .0
        self.rgt_outer.jnts[1]['rngmax'] = .8  # TODO change min-max to a tuple
        self.rgt_outer.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[2]['loc_pos'] = np.array([0, .0315, -.0041])  # passive
        self.rgt_outer.jnts[2]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[3]['loc_pos'] = np.array([0, .0061, .0471])
        self.rgt_outer.jnts[3]['rngmin'] = -.8757
        self.rgt_outer.jnts[3]['rngmax'] = .0  # TODO change min-max to a tuple
        self.rgt_outer.jnts[3]['loc_motionax'] = np.array([1, 0, 0])
        self.rgt_outer.jnts[4]['loc_pos'] = np.zeros(3)
        # https://github.com/Danfoa uses geometry instead of the dae mesh. The following coordiante is needed
        # self.rgt_outer.jnts[4]['loc_pos'] = np.array([0, -0.0220203446692936, .03242])
        # - rgt_inner
        self.rgt_inner = jl.JLChain(position=np.zeros(3), rotmat=np.eye(3), homeconf=np.zeros(1), name='rgt_inner')
        self.rgt_inner.jnts[1]['loc_pos'] = np.array([0, .0127, .06142])
        self.rgt_inner.jnts[1]['rngmin'] = .0
        self.rgt_inner.jnts[1]['rngmax'] = .8757  # TODO change min-max to a tuple
        self.rgt_inner.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        # links
        # - lft_outer
        self.lft_outer.lnks[0]['name'] = "robotiq85_gripper_base"
        self.lft_outer.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[0]['com'] = np.array([8.625e-08, -4.6583e-06, 0.03145])
        self.lft_outer.lnks[0]['mass'] = 0.22652
        self.lft_outer.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_base_link_cvt.stl")
        self.lft_outer.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.lft_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.lft_outer.lnks[1]['mass'] = 0.00853198276973456
        self.lft_outer.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.dae")
        self.lft_outer.lnks[1]['scale'] = [1e-3, 1e-3, 1e-3]
        self.lft_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.lft_outer.lnks[2]['name'] = "left_outer_finger"
        self.lft_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.lft_outer.lnks[2]['mass'] = 0.022614240507152
        self.lft_outer.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        self.lft_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[3]['name'] = "left_inner_finger"
        self.lft_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.lft_outer.lnks[3]['mass'] = 0.0104003125914103
        self.lft_outer.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_finger_cvt.stl")
        self.lft_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.lft_outer.lnks[4]['name'] = "left_inner_finger_pad"
        self.lft_outer.lnks[4]['loc_pos'] = np.zeros(3)
        self.lft_outer.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        self.lft_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        self.lft_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - lft_inner
        self.lft_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.lft_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.lft_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.lft_inner.lnks[1]['mass'] = 0.0271177346495152
        self.lft_inner.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        self.lft_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt_outer
        self.rgt_outer.lnks[1]['name'] = "left_outer_knuckle"
        self.rgt_outer.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[1]['com'] = np.array([-0.000200000000003065, 0.0199435877845359, 0.0292245259211331])
        self.rgt_outer.lnks[1]['mass'] = 0.00853198276973456
        self.rgt_outer.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_knuckle.dae")
        self.rgt_outer.lnks[1]['scale'] = [1e-3, 1e-3, 1e-3]
        self.rgt_outer.lnks[1]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        self.rgt_outer.lnks[2]['name'] = "left_outer_finger"
        self.rgt_outer.lnks[2]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[2]['com'] = np.array([0.00030115855001899, 0.0373907951953854, -0.0208027427000385])
        self.rgt_outer.lnks[2]['mass'] = 0.022614240507152
        self.rgt_outer.lnks[2]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_outer_finger_cvt.stl")
        self.rgt_outer.lnks[2]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[3]['name'] = "left_inner_finger"
        self.rgt_outer.lnks[3]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[3]['com'] = np.array([0.000299999999999317, 0.0160078233491243, -0.0136945669206257])
        self.rgt_outer.lnks[3]['mass'] = 0.0104003125914103
        self.rgt_outer.lnks[3]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_finger_cvt.stl")
        self.rgt_outer.lnks[3]['rgba'] = [.2, .2, .2, 1]
        self.rgt_outer.lnks[4]['name'] = "left_inner_finger_pad"
        self.rgt_outer.lnks[4]['loc_pos'] = np.zeros(3)
        self.rgt_outer.lnks[4]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_pad.dae")
        self.rgt_outer.lnks[4]['scale'] = [1e-3, 1e-3, 1e-3]
        self.rgt_outer.lnks[4]['rgba'] = [0.792156862745098, 0.819607843137255, 0.933333333333333, 1]
        # - rgt_inner
        self.rgt_inner.lnks[1]['name'] = "left_inner_knuckle"
        self.rgt_inner.lnks[1]['loc_pos'] = np.zeros(3)
        self.rgt_inner.lnks[1]['com'] = np.array([0.000123011831763771, 0.0507850843201817, 0.00103968640075166])
        self.rgt_inner.lnks[1]['mass'] = 0.0271177346495152
        self.rgt_inner.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_arg2f_85_inner_knuckle_cvt.stl")
        self.rgt_inner.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft_inner.reinitialize()
        self.lft_outer.reinitialize()
        self.rgt_outer.reinitialize()
        self.rgt_inner.reinitialize()

    def gen_stickmodel(self, togglejntscs=False, name='xarm_gripper_stickmodel'):
        stickmodel = gm.StaticGeometricModel(name=name)
        self.lft_outer.gen_stickmodel(togglejntscs=togglejntscs).attach_to(stickmodel)
        self.lft_inner.gen_stickmodel(togglejntscs=togglejntscs).attach_to(stickmodel)
        self.rgt_outer.gen_stickmodel(togglejntscs=togglejntscs).attach_to(stickmodel)
        self.rgt_inner.gen_stickmodel(togglejntscs=togglejntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self, name='xarm_gripper_meshmodel'):
        stickmodel = gm.StaticGeometricModel(name=name)
        self.lft_outer.gen_meshmodel().attach_to(stickmodel)
        self.lft_inner.gen_meshmodel().attach_to(stickmodel)
        self.rgt_outer.gen_meshmodel().attach_to(stickmodel)
        self.rgt_inner.gen_meshmodel().attach_to(stickmodel)
        return stickmodel

    def fk(self, angle):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        if self.lft_outer.jnts[1]['rngmin'] <= angle <= self.lft_outer.jnts[1]['rngmax']:
            self.lft_outer.jnts[1]['motion_val'] = angle
            self.lft_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.lft_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.rgt_outer.jnts[3]['motion_val'] = -self.lft_outer.jnts[1]['motion_val']
            self.rgt_inner.jnts[1]['motion_val'] = self.lft_outer.jnts[1]['motion_val']
            self.lft_outer.fk()
            self.lft_inner.fk()
            self.rgt_outer.fk()
            self.rgt_inner.fk()
        else:
            raise ValueError("The angle parameter is out of range!")


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[.5,.5,.5], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = Robotiq85()
    grpr.fk(.8)
    grpr.gen_meshmodel().attach_to(base)
    grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    base.run()

    base = wd.World(campos=[.5,.5,.5], lookatpos=[0, 0, 0])
    model = gm.GeometricModel("./meshes/robotiq_arg2f_85_pad.dae")
    model.set_scale([1e-3, 1e-3, 1e-3])
    model.attach_to(base)
    gm.gen_frame().attach_to(base)
    base.run()
