import os
import numpy as np
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import basis.robotmath as rm


class RobotiqHE(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='robotiqhe'):
        this_dir, this_filename = os.path.split(__file__)
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        # joints
        # - coupling - .011 coupling by default
        self.coupling = jl.JLChain(pos=self.pos, rotmat=self.rotmat, homeconf=np.zeros(0), name='coupling')
        self.coupling.jnts[1]['loc_pos'] = np.array([0, 0, .011])
        self.coupling.lnks[0]['name'] = self.name+"_coupling"
        # self.coupling.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "robotiq_gripper_coupling.stl")
        self.coupling.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.coupling.reinitialize()
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft
        self.lft = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='base_lft_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([-.025, .0, .11])
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['rngmin'] = .0
        self.lft.jnts[1]['rngmax'] = .025
        self.lft.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "base_cvt.stl")
        self.lft.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "finger1_cvt.stl")
        self.lft.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # - rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos, rotmat=cpl_end_rotmat, homeconf=np.zeros(1), name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([.025, .0, .11])
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['rngmin'] = .0
        self.rgt.jnts[1]['rngmax'] = .025  # TODO change min-max to a tuple
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "finger2_cvt.stl")
        self.rgt.lnks[1]['rgba'] = [.5, .5, .5, 1]
        # reinitialize
        self.lft.reinitialize()
        self.rgt.reinitialize()

    def fix_to(self, pos, rotmat, jawwidth=None):
        self.pos = pos
        self.rotmat = rotmat
        if jawwidth is not None:
            side_jawwidth = (.05 - jawwidth) / 2.0
            if 0 <= side_jawwidth <= .025:
                self.lft.jnts[1]['motion_val'] = side_jawwidth;
                self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            else:
                raise ValueError("The angle parameter is out of range!")
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, jawwidth):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: angle, radian
        """
        side_jawwidth = (.05 - jawwidth) / 2.0
        if 0 <= side_jawwidth <= .025:
            self.lft.jnts[1]['motion_val'] = side_jawwidth;
            self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
            self.lft.fk()
            self.rgt.fk()
        else:
            raise ValueError("The angle parameter is out of range!")

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm_gripper_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.coupling.gen_stickmodel(tcp_loc_pos=None,
                                     tcp_loc_rotmat=None,
                                     toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.lft.gen_stickmodel(tcp_jntid=tcp_jntid,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.rgt.gen_stickmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      name='xarm_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.lft.gen_meshmodel(tcp_jntid=tcp_jntid,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[.5, .5, .5], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    # for angle in np.linspace(0, .85, 8):
    #     grpr = Robotiq85()
    #     grpr.fk(angle)
    #     grpr.gen_meshmodel().attach_to(base)
    grpr = RobotiqHE()
    grpr.fk(.05)
    grpr.gen_meshmodel().attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)
    base.run()
