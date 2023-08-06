import os
import math
import numpy as np
import modeling.collision_model as cm
import modeling.model_collection as mc
import robot_sim._kinematics.jlchain as jl
import robot_sim.manipulators.rs007l.rs007l as manipulator
import robot_sim.robots.robot_interface as ri


class KHI_BLQC(ri.RobotInterface):
    """
    author: weiwei
    date: 20230826, Toyonaka
    """

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="khi_g", enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        # arm
        self.arm = manipulator.RS007L(pos=pos,
                                      rotmat=rotmat,
                                      homeconf=np.zeros(6),
                                      name='rs007l', enable_cc=False)
        # tool changer
        self.tc_master = jl.JLChain(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'],
                                    homeconf=np.zeros(0), name='tc_master')
        self.tc_master.lnks[0]['name'] = "tc_master"
        self.tc_master.jnts[-1]['loc_pos'] = np.array([0, 0, .0315])
        self.tc_master.lnks[0]['collision_model'] = cm.gen_stick(self.tc_master.jnts[0]['loc_pos'],
                                                                 # TODO: change to combined model, 20230806
                                                                 self.tc_master.jnts[-1]['loc_pos'],
                                                                 thickness=0.05,
                                                                 # rgba=[.2, .2, .2, 1], rgb will be overwritten
                                                                 type='rect',
                                                                 sections=36)
        self.tc_master.reinitialize()
        # tool center point
        self.arm.jlc.tcp_jnt_id = -1
        self.arm.jlc.tcp_loc_pos = self.tc_master.jnts[-1]['loc_pos']
        self.arm.jlc.tcp_loc_rotmat = self.tc_master.jnts[-1]['loc_rotmat']
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self._is_tool_attached = False
        # a list of detailed information about objects in hand, see CollisionChecker.add_objinhnd
        self.oih_infos = []
        # collision detection
        if enable_cc:
            self.enable_cc()
        # component map
        self.manipulator_dict['arm'] = self.arm
        self.manipulator_dict['tc_master'] = self.tc_master
        self.hnd_dict['arm'] = self.arm
        self.hnd_dict['tc_master'] = self.tc_master

    @property
    def is_tool_attached(self):
        return self._is_tool_attached

    def enable_cc(self):
        pass
        # TODO when pose is changed, oih info goes wrong
        # super().enable_cc()
        # self.cc.add_cdlnks(self.base_plate, [0])
        # self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        # self.cc.add_cdlnks(self.tc_master.jlc, [0, 1, 2])
        # activelist = [self.base_plate.lnks[0],
        #               self.arm.lnks[0],
        #               self.arm.lnks[1],
        #               self.arm.lnks[2],
        #               self.arm.lnks[3],
        #               self.arm.lnks[4],
        #               self.arm.lnks[5],
        #               self.arm.lnks[6],
        #               self.tc_master.jlc.lnks[0],
        #               self.tc_master.jlc.lnks[1],
        #               self.tc_master.jlc.lnks[2]]
        # self.cc.set_active_cdlnks(activelist)
        # fromlist = [self.base_plate.lnks[0],
        #             self.arm.lnks[0],
        #             self.arm.lnks[1]]
        # intolist = [self.arm.lnks[3]]
        # self.cc.set_cdpair(fromlist, intolist)
        # fromlist = [self.base_plate.lnks[0],
        #             self.arm.lnks[1]]
        # intolist = [self.tc_master.jlc.lnks[0],
        #             self.tc_master.jlc.lnks[1],
        #             self.tc_master.jlc.lnks[2]]
        # self.cc.set_cdpair(fromlist, intolist)
        # # TODO is the following update needed?
        # for oih_info in self.oih_infos:
        #     objcm = oih_info['collision_model']
        #     self.hold(objcm)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.base_plate.fix_to(pos=pos, rotmat=rotmat)
        self.arm.fix_to(pos=self.base_plate.jnts[-1]['gl_posq'], rotmat=self.base_plate.jnts[-1]['gl_rotmatq'])
        self.tc_master.fix_to(pos=self.arm.jnts[-1]['gl_posq'], rotmat=self.arm.jnts[-1]['gl_rotmatq'])
        # update objects in hand if available
        for obj_info in self.oih_infos:
            gl_pos, gl_rotmat = self.arm.cvt_loc_tcp_to_gl(obj_info['rel_pos'], obj_info['rel_rotmat'])
            obj_info['gl_pos'] = gl_pos
            obj_info['gl_rotmat'] = gl_rotmat

    def fk(self, component_name='arm', jnt_values=np.zeros(6)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """

        def update_oih(component_name='arm'):
            for obj_info in self.oih_infos:
                gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(component_name, obj_info['rel_pos'], obj_info['rel_rotmat'])
                obj_info['gl_pos'] = gl_pos
                obj_info['gl_rotmat'] = gl_rotmat

        def update_component(component_name, jnt_values):
            status = self.manipulator_dict[component_name].fk(jnt_values=jnt_values)
            self.tc_master_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq'])
            update_oih(component_name=component_name)
            return status

        if component_name in self.manipulator_dict:
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != 6:
                raise ValueError("An 1x6 npdarray must be specified to move the arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not supported!")

    def get_jnt_values(self, component_name):
        if component_name in self.manipulator_dict:
            return self.manipulator_dict[component_name].get_jnt_values()
        else:
            raise ValueError("The given component name is not supported!")

    def get_gl_tcp(self, manipulator_name='arm'):
        return self.manipulator_dict[manipulator_name].get_gl_tcp()

    def rand_conf(self, component_name):
        if component_name in self.manipulator_dict:
            return super().rand_conf(component_name)
        else:
            raise NotImplementedError

    def jaw_to(self, hnd_name='hnd', jawwidth=0.0):
        raise Exception("This robot has a single-contact end effector (screw driver).")

    def attach_tool(self, hnd):
        """
        :param hnd: an instance of robot_sim.end_effectors
        :return:
        """
        if self._is_tool_attached:
            raise Exception("A tool has been attached!")
        self.arm.jlc.tcp_loc_pos = self.tc_master.jnts[-1]['gl_posq'] + hnd.jaw_center_pos
        self.arm.jlc.tcp_loc_rotmat = self.tc_master.jnts[-1]['gl_rotmatq'] + hnd.jaw_center_rotmat
        self.manipulator_dict['hnd'] = self.hnd
        self.hnd_dict['hnd'] = self.hnd
        self._is_tool_attached = True

    def detach_tool(self):
        """
        there is no need to specify a tool instance since only the currently attached tool can be detached
        :return:
        """
        if not self._is_tool_attached:
            raise Exception("Cannot detach a tool since nothing is attached!")
        self.arm.jlc.tcp_loc_pos = self.tc_master.jnts[-1]['gl_posq']
        self.arm.jlc.tcp_loc_rotmat = self.tc_master.jnts[-1]['gl_rotmatq']
        self.manipulator_dict.pop('hnd')
        self.hnd_dict.pop('hnd')
        self._is_tool_attached = False

    def get_oih_list(self):
        return_list = []
        for obj_info in self.oih_infos:
            objcm = obj_info['collision_model']
            objcm.set_pos(obj_info['gl_pos'])
            objcm.set_rotmat(obj_info['gl_rotmat'])
            return_list.append(objcm)
        return return_list

    def gen_stickmodel(self,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='khi_g_stickmodel'):
        stickmodel = mc.ModelCollection(name=name)
        self.arm.gen_stickmodel(tcp_jnt_id=tcp_jnt_id,
                                tcp_loc_pos=tcp_loc_pos,
                                tcp_loc_rotmat=tcp_loc_rotmat,
                                toggle_tcpcs=toggle_tcpcs,
                                toggle_jntscs=toggle_jntscs,
                                toggle_connjnt=toggle_connjnt).attach_to(stickmodel)
        self.tc_master.gen_stickmodel(toggle_tcpcs=False,
                                      toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        if self.hnd_dict.get('hnd'):
            self.hnd_dict['hnd'].gen_stickmodel(toggle_tcpcs=False,
                                                toggle_jntscs=toggle_jntscs).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='khi_g_meshmodel'):
        meshmodel = mc.ModelCollection(name=name)
        self.arm.gen_meshmodel(tcp_jnt_id=tcp_jnt_id,
                               tcp_loc_pos=tcp_loc_pos,
                               tcp_loc_rotmat=tcp_loc_rotmat,
                               toggle_tcpcs=toggle_tcpcs,
                               toggle_jntscs=toggle_jntscs,
                               rgba=rgba).attach_to(meshmodel)
        self.tc_master.gen_meshmodel(toggle_tcpcs=False,
                                     toggle_jntscs=toggle_jntscs,
                                     rgba=rgba).attach_to(meshmodel)
        if self.hnd_dict.get('hnd'):
            self.hnd_dict['hnd'].gen_meshmodel(toggle_tcpcs=False,
                                               toggle_jntscs=toggle_jntscs,
                                               rgba=rgba).attach_to(meshmodel)
            for obj_info in self.oih_infos:
                objcm = obj_info['collision_model']
                objcm.set_pos(obj_info['gl_pos'])
                objcm.set_rotmat(obj_info['gl_rotmat'])
                objcm.copy().attach_to(meshmodel)
        return meshmodel

    def ik(self,
           component_name: str = "arm",
           tgt_pos=np.zeros(3),
           tgt_rotmat=np.eye(3),
           seed_jnt_values=None,
           max_niter=200,
           tcp_jnt_id=None,
           tcp_loc_pos=None,
           tcp_loc_rotmat=None,
           local_minima: str = "end",
           toggle_debug=False):
        return self.manipulator_dict[component_name].analytical_ik(tgt_pos,
                                                                   tgt_rotmat,
                                                                   tcp_loc_pos=tcp_loc_pos,
                                                                   tcp_loc_rotmat=tcp_loc_rotmat)


if __name__ == '__main__':
    import time
    import basis.robot_math as rm
    import visualization.panda.world as wd
    import modeling.geometric_model as gm

    base = wd.World(cam_pos=[1.7, 1.7, 1.7], lookat_pos=[0, 0, .3])

    gm.gen_frame().attach_to(base)
    robot_s = KHI_BLQC(enable_cc=True)
    # robot_s.jaw_to(.02)
    robot_s.gen_meshmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    # robot_s.gen_meshmodel(toggle_tcpcs=False, toggle_jntscs=False).attach_to(base)
    robot_s.gen_stickmodel(toggle_tcpcs=True, toggle_jntscs=True).attach_to(base)
    base.run()
    tgt_pos = np.array([.45, .2, .35])
    tgt_rotmat = rm.rotmat_from_axangle([0, 1, 0], math.pi * 2 / 3)
    gm.gen_frame(pos=tgt_pos, rotmat=tgt_rotmat).attach_to(base)
    # base.run()
    component_name = 'arm'
    jnt_values = robot_s.ik(component_name, tgt_pos, tgt_rotmat)
    robot_s.fk(component_name, jnt_values=jnt_values)
    robot_s_meshmodel = robot_s.gen_meshmodel(toggle_tcpcs=True)
    robot_s_meshmodel.attach_to(base)
    # robot_s.show_cdprimit()
    robot_s.gen_stickmodel().attach_to(base)
    # tic = time.time()
    # result = robot_s.is_collided()
    # toc = time.time()
    # print(result, toc - tic)
    base.run()
