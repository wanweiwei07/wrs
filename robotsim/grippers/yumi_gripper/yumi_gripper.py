import os
import math
import numpy as np
import modeling.modelcollection as mc
import robotsim._kinematics.jlchain as jl
import basis.robotmath as rm
import robotsim.grippers.grippers as gp

class YumiGripper(gp.Gripper):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='yumi_gripper'):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        # - lft
        self.lft = jl.JLChain(pos=cpl_end_pos,
                              rotmat=cpl_end_rotmat,
                              homeconf=np.zeros(1),
                              name='base_lft_finger')
        self.lft.jnts[1]['loc_pos'] = np.array([0, 0.0065, 0.0837])
        self.lft.jnts[1]['loc_rotmat'] = rm.rotmat_from_euler(0, 0, math.pi)
        self.lft.jnts[1]['type'] = 'prismatic'
        self.lft.jnts[1]['rngmin'] = .0
        self.lft.jnts[1]['rngmax'] = .025
        self.lft.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.lft.lnks[0]['name'] = "base"
        self.lft.lnks[0]['loc_pos'] = np.zeros(3)
        self.lft.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "base.stl")
        self.lft.lnks[0]['rgba'] = [.5, .5, .5, 1]
        self.lft.lnks[1]['name'] = "finger1"
        self.lft.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.lft.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # - rgt
        self.rgt = jl.JLChain(pos=cpl_end_pos,
                              rotmat=cpl_end_rotmat,
                              homeconf=np.zeros(1),
                              name='rgt_finger')
        self.rgt.jnts[1]['loc_pos'] = np.array([0, -0.0065, 0.0837])
        self.rgt.jnts[1]['type'] = 'prismatic'
        self.rgt.jnts[1]['rngmin'] = .0
        self.rgt.jnts[1]['rngmax'] = .025  # TODO change min-max to a tuple
        self.rgt.jnts[1]['loc_motionax'] = np.array([-1, 0, 0])
        self.rgt.lnks[1]['name'] = "finger2"
        self.rgt.lnks[1]['meshfile'] = os.path.join(this_dir, "meshes", "finger.stl")
        self.rgt.lnks[1]['rgba'] = [.2, .2, .2, 1]
        # reinitialize
        self.lft.reinitialize()
        self.rgt.reinitialize()
        # disable the localcc of each the links
        self.lft.disable_localcc()
        self.rgt.disable_localcc()
        # collision detection
        self.cc.add_cdlnks(self.lft, [0, 1])
        self.cc.add_cdlnks(self.rgt, [1])
        activelist = [self.lft.lnks[0],
                      self.lft.lnks[1],
                      self.rgt.lnks[1]]
        self.cc.set_active_cdlnks(activelist)

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        # object in hand do not update by itself
        is_fk_updated = self.lft.is_fk_updated
        return self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list,
                                   need_update=is_fk_updated)

    def is_mesh_collided(self, objcm_list=[]):
        hnd_objcm_list = [self.lft.lnks[0]['collisionmodel'],
                          self.lft.lnks[1]['collisionmodel'],
                          self.rgt.lnks[1]['collisionmodel']]
        for objcm in objcm_list:
            if objcm.is_mcdwith(hnd_objcm_list):
                return True
        return False

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.coupling.fix_to(self.pos, self.rotmat)
        cpl_end_pos = self.coupling.jnts[-1]['gl_posq']
        cpl_end_rotmat = self.coupling.jnts[-1]['gl_rotmatq']
        self.lft.fix_to(cpl_end_pos, cpl_end_rotmat)
        self.rgt.fix_to(cpl_end_pos, cpl_end_rotmat)

    def fk(self, motion_val):
        """
        lft_outer is the only active joint, all others mimic this one
        :param: motion_val, meter or radian
        """
        if motion_val > .05:
            raise ValueError("The angle parameter is out of range!")
        side_motion_val = (.05 - motion_val) / 2.0
        self.lft.jnts[1]['motion_val'] = side_motion_val
        self.rgt.jnts[1]['motion_val'] = self.lft.jnts[1]['motion_val']
        self.lft.fk()
        self.rgt.fk()

    def jaw_to(self, jawwidth):
        self.fk(jawwidth=jawwidth)

    def show_cdprimit(self):
        is_fk_updated = self.lft.is_fk_updated
        self.cc.show_cdprimit(need_update=is_fk_updated)

    def show_cdmesh(self):
        pass

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='yumi_gripper_stickmodel'):
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
                      rgba=None,
                      name='yumi_gripper_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.coupling.gen_meshmodel(tcp_loc_pos=None,
                                    tcp_loc_rotmat=None,
                                    toggle_tcpcs=False,
                                    toggle_jntscs=toggle_jntscs,
                                    rgba=rgba).attach_to(meshmodel)
        self.lft.gen_meshmodel(tcp_jntid=tcp_jntid,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.rgt.gen_meshmodel(tcp_loc_pos=None,
                               tcp_loc_rotmat=None,
                               toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    import copy
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(campos=[.5, .5, .5], lookatpos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    grpr = YumiGripper()
    grpr.fk(.05)
    grpr.gen_meshmodel(rgba=[0, .5, 0, .5]).attach_to(base)
    # grpr.gen_stickmodel(togglejntscs=False).attach_to(base)
    grpr.fix_to(pos=np.array([0, .3, .2]), rotmat=rm.rotmat_from_axangle([1, 0, 0], .05))
    grpr.gen_meshmodel().attach_to(base)

    grpr2 = grpr.copy()
    grpr2.fix_to(pos=np.array([.3,.3,.2]), rotmat=rm.rotmat_from_axangle([0,1,0],.01))
    model = grpr2.gen_meshmodel(rgba=[0.5, .5, 0, .5])
    model.attach_to(base)
    grpr2.show_cdprimit()
    model.show_cdmesh()
    base.run()
