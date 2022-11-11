"""
Simulation for the XArm Lite 6 With the WRS gripper
Author: Chen Hao (chen960216@gmail.com), 20220925, osaka
Reference: The code is implemented referring to the 'robot_sim/robots/ur3e_dual/'
"""
import os
import numpy as np
import basis.robot_math as rm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
from robot_sim.manipulators.xarm_lite6 import XArmLite6
from robot_sim.end_effectors.gripper.lite6_wrs_gripper import Lite6WRSGripper
import robot_sim.robots.robot_interface as ri


class XArmLite6WRSGripper(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.body = jl.JLChain(pos=pos, rotmat=rotmat, homeconf=np.zeros(0), name='lft_body_jl')
        self.arm = XArmLite6(pos=self.body.jnts[-1]['gl_posq'],
                             rotmat=self.body.jnts[-1]['gl_rotmatq'],
                             enable_cc=False)
        arm_tcp_rotmat = self.arm.jnts[-1]['gl_rotmatq']
        self.hnd = Lite6WRSGripper(pos=self.arm.jnts[-1]['gl_posq'],
                                   rotmat=rm.rotmat_from_axangle(arm_tcp_rotmat[:3, 2], np.radians(90)).dot(
                                       arm_tcp_rotmat),
                                   enable_cc=False)

        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.hnd.jaw_center_rotmat
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['hnd'] = self.arm  # specify which hand is a gripper installed to
        self.hnd_dict['hnd'] = self.hnd
        self.hnd_dict['arm'] = self.hnd

    def enable_cc(self):
        super().enable_cc()
        # raise NotImplementedError
        # self.cc.add_cdlnks(self.body, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.lft, [0, 1])
        self.cc.add_cdlnks(self.hnd.rgt, [1])
        # lnks used for cd with external stationary objects
        activelist = [self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.lft.lnks[0],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1], ]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        # fromlist = [self.body.lnks[0],
        #             self.arm.lnks[1], ]
        fromlist = [self.arm.lnks[0],
                    self.arm.lnks[1], ]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1], ]
        self.cc.set_cdpair(fromlist, intolist)
        # lnks used for arm-body collision detection -- extra
        # fromlist = [self.body.lnks[0]]  # body
        # intolist = [self.arm.lnks[2], ]
        # self.cc.set_cdpair(fromlist, intolist)
        # lnks used for in-arm collision detection
        fromlist = [self.arm.lnks[2]]
        intolist = [self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.hnd.lft.lnks[0],
                    self.hnd.lft.lnks[1],
                    self.hnd.rgt.lnks[1]]
        self.cc.set_cdpair(fromlist, intolist)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.body.fix_to(self.pos, self.rotmat)
        self.arm.fix_to(pos=self.body.jnts[-1]['gl_posq'], rotmat=self.body.jnts[-1]['gl_rotmatq'])
        arm_tcp_rotmat = self.arm.jnts[-1]['gl_rotmatq']
        self.hnd.fix_to(pos=self.arm.jnts[-1]['gl_posq'],
                        rotmat=rm.rotmat_from_axangle(arm_tcp_rotmat[:3, 2], np.radians(90)).dot(arm_tcp_rotmat))
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name, jnt_values):
        """
        :param jnt_values: 1x6 or 1x12 nparray
        :hnd_name 'lft_arm', 'rgt_arm', 'both_arm'
        :param component_name:
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            # inline function for update objects in hand
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            arm_tcp_rotmat = self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq']
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=rm.rotmat_from_axangle(arm_tcp_rotmat[:3, 2], np.radians(90)).dot(arm_tcp_rotmat))
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine length
        if component_name == 'arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != self.arm.ndof:
                raise ValueError(f"An 1x{self.arm.ndof} npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not available!")

    def rand_conf(self, component_name):
        """
        override robot_interface.rand_conf
        :param component_name:
        :return:
        author: weiwei
        date: 20210406
        """
        if component_name == 'arm':
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='xarm_lite6_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.body.gen_stickmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcpcs=False,
                                 toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.hnd.gen_stickmodel(toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='xarm_lite6_meshmodel'):
        mm_collection = mc.ModelCollection(name=name)
        self.body.gen_meshmodel(tcp_loc_pos=None,
                                tcp_loc_rotmat=None,
                                toggle_tcpcs=False,
                                toggle_jntscs=toggle_jntscs,
                                rgba=rgba).attach_to(mm_collection)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(mm_collection)
        self.hnd.gen_meshmodel(toggle_tcpcs=False,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(mm_collection)
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            objcm.copy().attach_to(mm_collection)
        return mm_collection


if __name__ == '__main__':
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, 0])
    gm.gen_frame().attach_to(base)
    xarm = XArmLite6WRSGripper(enable_cc=True)
    rand_conf = xarm.rand_conf(component_name='arm')
    xarm.fk('arm', rand_conf)
    xarm_meshmodel = xarm.gen_meshmodel(toggle_tcpcs=False)
    xarm_meshmodel.attach_to(base)
    # xarm_meshmodel.show_cdprimit()
    # xarm.gen_stickmodel().attach_to(base)
    print("Is self collided?", xarm.is_collided())
    base.run()
