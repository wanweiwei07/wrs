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
from robot_sim.end_effectors.gripper.lite6_wrs_gripper.lite6_wrs_gripper_v2 import Lite6WRSGripper2
import robot_sim.robots.system_interface as ri


class XArmLite6WRSGripper(ri.RobotInterface):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name='xarm_lite6', enable_cc=True):
        super().__init__(pos=pos, rotmat=rotmat, name=name)
        this_dir, this_filename = os.path.split(__file__)
        # left side
        self.body = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(0), name='body_jl')
        self.body.lnks[0]['name'] = "xarm_stand"
        self.body.lnks[0]['pos_in_loc_tcp'] = np.array([0, 0, 0])
        self.body.lnks[0]['rgba'] = [.55, .55, .55, 1.0]
        self.body.lnks[0]['mesh_file'] = os.path.join(this_dir, "meshes", "xarm_lite6_stand.stl")
        self.body.finalize()
        self.arm = XArmLite6(pos=self.body.jnts[-1]['gl_posq'],
                             rotmat=self.body.jnts[-1]['gl_rotmatq'],
                             enable_cc=False)
        arm_tcp_rotmat = self.arm.jnts[-1]['gl_rotmatq']
        self.hnd = Lite6WRSGripper2(pos=self.arm.jnts[-1]['gl_posq'], rotmat=arm_tcp_rotmat,
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
        self.cc.add_cdlnks(self.body, [0])
        self.cc.add_cdlnks(self.arm, [0, 1, 2, 3, 4, 5, 6])
        self.cc.add_cdlnks(self.hnd.body, [0, 1])
        self.cc.add_cdlnks(self.hnd.lft, [1])
        self.cc.add_cdlnks(self.hnd.rgt, [1])
        # lnks used for cd with external stationary objects
        activelist = [self.arm.lnks[2],
                      self.arm.lnks[3],
                      self.arm.lnks[4],
                      self.arm.lnks[5],
                      self.arm.lnks[6],
                      self.hnd.body.lnks[0],
                      self.hnd.body.lnks[1],
                      self.hnd.lft.lnks[1],
                      self.hnd.rgt.lnks[1], ]
        self.cc.set_active_cdlnks(activelist)
        # lnks used for arm-body collision detection
        # fromlist = [self.body.lnks[0],
        #             self.arm.lnks[1], ]
        fromlist = [self.body.lnks[0],
                    self.arm.lnks[0],
                    self.arm.lnks[1], ]
        intolist = [self.arm.lnks[3],
                    self.arm.lnks[4],
                    self.arm.lnks[5],
                    self.arm.lnks[6],
                    self.hnd.body.lnks[0],
                    self.hnd.body.lnks[1],
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
                    self.hnd.body.lnks[0],
                    self.hnd.body.lnks[1],
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
                        rotmat=arm_tcp_rotmat)
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
            status = self.manipulator_dict[component_name].fk(joint_values=jnt_values)
            arm_tcp_rotmat = self.manipulator_dict[component_name].jnts[-1]['gl_rotmatq']
            self.hnd_dict[component_name].fix_to(
                pos=self.manipulator_dict[component_name].jnts[-1]['gl_posq'],
                rotmat=arm_tcp_rotmat)
            update_oih(component_name=component_name)
            return status

        super().fk(component_name, jnt_values)
        # examine axis_length
        if component_name == 'arm':
            if not isinstance(jnt_values, np.ndarray) or jnt_values.size != self.arm.ndof:
                raise ValueError(f"An 1x{self.arm.ndof} npdarray must be specified to move a single arm!")
            return update_component(component_name, jnt_values)
        else:
            raise ValueError("The given component name is not available!")


    def analytical_ik(self,
                      tgt_pos=np.zeros(3),
                      tgt_rotmat=np.eye(3),
                      seed_jnt_values=None,
                      return_all_solutions=False):
        """
        :param tgt_pos:
        :param tgt_rotmat:
        :param seed_jnt_values:
        :param return_all_solutions:
        :return:
        Modified D-H from base to tcp
        T01 = sympy.Matrix([[c1, -s1, 0, 0],
                            [s1, c1, 0, 0],
                            [0, 0, 1, d1],
                            [0, 0, 0, 1]])
        T12 = sympy.Matrix([[s2, c2, 0, 0],
                            [0, 0, 1, 0],
                            [c2, -s2, 0, 0],
                            [0, 0, 0, 1]])
        T23 = sympy.Matrix([[s3, c3, 0, a3],
                            [c3, -s3, 0, 0],
                            [0, 0, -1, 0],
                            [0, 0, 0, 1]])
        T34 = sympy.Matrix([[c4, -s4, 0, a4],
                            [0, 0, -1, -d4],
                            [s4, c4, 0, 0],
                            [0, 0, 0, 1]])
        T45 = sympy.Matrix([[c5, -s5, 0, 0],
                            [0, 0, -1, 0],
                            [s5, c5, 0, 0],
                            [0, 0, 0, 1]])
        T56 = sympy.Matrix([[c6, -s6, 0, 0],
                            [0, 0, 1, 0],
                            [-s6, -c6, 0, 0],
                            [0, 0, 0, 1]])
        T6t = sympy.Matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, d_hnd + d6],
                            [0, 0, 0, 1]])
        T0t=T01*T12*T23*T34*T45*T56*T6t
        """
        # lengths of link
        d1, d4, d6, dtcp = .2433, .2276, .0615, .175
        a3, a4 = .2, .087
        # global position of joint 1
        pos_1 = np.array([0, 0, d1])
        # Joint 1
        pos_w = tgt_pos - (d6 + dtcp) * tgt_rotmat[:, 2]
        theta1 = np.arctan2(pos_w[1], pos_w[0])
        # Joint 3
        d1w = np.sum((pos_w-pos_1)**2)
        num3_1 = 2*a3*a4
        num3_2 = - a3**4 + 2*a3**2*a4**2 + 2*a3**2*d1w + 2*a3**2*d4**2 - a4**4 + 2*a4**2*d1w - 2*a4**2*d4**2\
                 - d1w**2 + 2*d1w*d4**2 - d4**4
        den3 = a3**2 + 2*a3*d4 + a4**2 - d1w + d4**2
        theta3_list = [-2*np.arctan2((num3_1-np.sqrt(num3_2)), den3),
                       -2*np.arctan2((num3_1+np.sqrt(num3_2)), den3)]
        theta2_4_5_6_list = []
        for theta3 in theta3_list:
            # Joint 2
            dxy = np.sqrt(pos_w[0]**2 + pos_w[1]**2)
            z_w = pos_w[2]
            num_c2 = -a3*d1 + a3*z_w - a4*d1*np.sin(theta3) + a4*dxy*np.cos(theta3) + a4*z_w*np.sin(theta3) + \
                     d1*d4*np.cos(theta3) + d4*dxy*np.sin(theta3) - d4*z_w*np.cos(theta3)
            num_s2 = a3*dxy + a4*d1*np.cos(theta3) + a4*dxy*np.sin(theta3) - a4*z_w*np.cos(theta3) + \
                     d1*d4*np.sin(theta3) - d4*dxy*np.cos(theta3) - d4*z_w*np.sin(theta3)
            den_cs2 = a3**2 + 2*a3*a4*np.sin(theta3) - 2*a3*d4*np.cos(theta3) + a4**2*np.sin(theta3)**2 + \
                      a4**2*np.cos(theta3)**2 + d4**2*np.sin(theta3)**2 + d4**2*np.cos(theta3)**2

            if den_cs2 < 0:
                num_s2 = -num_s2
                num_c2 = -num_c2
            theta2 = np.arctan2(num_s2, num_c2)
            # Joint 4,5,6
            U06 = rm.homomat_from_posrot(pos=pos_w, rotmat=tgt_rotmat)
            s1, c1, s2, c2, s3, c3 = np.sin(theta1), np.cos(theta1), np.sin(theta2), np.cos(theta2), np.sin(theta3), np.cos(theta3)
            T03 = np.array([[s2*s3*c1 + c1*c2*c3, s2*c1*c3 - s3*c1*c2, s1, a3*s2*c1],
                            [s1*s2*s3 + s1*c2*c3, s1*s2*c3 - s1*s3*c2, -c1, a3*s1*s2],
                            [-s2*c3 + s3*c2, s2*s3 + c2*c3, 0, a3*c2 + d1],
                            [0, 0, 0, 1]])
            """
            U36 = 
            [[-sin(θ4)*sin(θ6) + cos(θ4)*cos(θ5)*cos(θ6), -sin(θ4)*cos(θ6) - sin(θ6)*cos(θ4)*cos(θ5), -sin(θ5)*cos(θ4), a4], 
             [-sin(θ5)*cos(θ6), sin(θ5)*sin(θ6), -cos(θ5), -d4], 
             [sin(θ4)*cos(θ5)*cos(θ6) + sin(θ6)*cos(θ4), -sin(θ4)*sin(θ6)*cos(θ5) + cos(θ4)*cos(θ6), -sin(θ4)*sin(θ5), 0], 
             [0, 0, 0, 1]]
            """
            U36 = (np.linalg.inv(T03)).dot(U06)
            theta6 = -np.arctan2(U36[1, 1], U36[1, 0])
            theta4 = np.arctan2(U36[2, 2], U36[0, 2])
            c5 = -U36[1, 2]
            if abs(U36[1, 0] / np.cos(theta6)) < abs(U36[1, 1] / np.sin(theta6)):
                s5 = -U36[1, 0] / np.cos(theta6)
            else:
                s5 = U36[1, 1] / np.sin(theta6)
            theta5 = np.arctan2(s5, c5)

            theta2_4_5_6_list.append((theta2, theta4, theta5, theta6))
        # Adapt to joint range
        if theta3_list[0] < .0:
            theta3_list[0] = 2.*np.pi + theta3_list[0]
        if theta3_list[1] < .0:
            theta3_list[1] = 2.*np.pi + theta3_list[1]
        jnt_values_list = [np.array([theta1, j2, j3, j4, j5, j6]) for j3, (j2, j4, j5, j6) in zip(theta3_list, theta2_4_5_6_list)]
        if not ((self.arm.jnts[3]['motion_rng'][0] <= jnt_values_list[1][2] <= self.arm.jnts[3]['motion_rng'][1]) and
                (self.arm.jnts[4]['motion_rng'][0] <= jnt_values_list[1][3] <= self.arm.jnts[4]['motion_rng'][1]) and
                (self.arm.jnts[5]['motion_rng'][0] <= jnt_values_list[1][4] <= self.arm.jnts[5]['motion_rng'][1])):
            jnt_values_list.pop(1)
        if not ((self.arm.jnts[3]['motion_rng'][0] <= jnt_values_list[0][2] <= self.arm.jnts[3]['motion_rng'][1]) and
                (self.arm.jnts[4]['motion_rng'][0] <= jnt_values_list[0][3] <= self.arm.jnts[4]['motion_rng'][1]) and
                (self.arm.jnts[5]['motion_rng'][0] <= jnt_values_list[0][4] <= self.arm.jnts[5]['motion_rng'][1])):
            jnt_values_list.pop(0)
        if len(jnt_values_list) == 0:
            return None
        if return_all_solutions is True:
            return jnt_values_list
        if len(jnt_values_list) == 1:
            return jnt_values_list[0]
        # return joint values close to seed_jnt_vals
        seed_jnt_values = np.zeros(6) if seed_jnt_values is None else seed_jnt_values
        if np.linalg.norm(jnt_values_list[0] - seed_jnt_values) < np.linalg.norm(jnt_values_list[1] - seed_jnt_values):
            return jnt_values_list[0]
        else:
            return jnt_values_list[1]

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

    def hold(self, hnd_name, objcm, jaw_width=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param hnd_name:
        :param jaw_width:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jaw_width is not None:
            self.hnd_dict[hnd_name].jaw_to(jaw_width)
        rel_pos, rel_rotmat = self.manipulator_dict[hnd_name].cvt_gl_to_loc_tcp(objcm.get_pos(), objcm.get_rotmat())
        into_list = [self.arm.lnks[0],
                     self.arm.lnks[1],
                     self.arm.lnks[2],
                     self.arm.lnks[3],
                     self.arm.lnks[4],
                     self.arm.lnks[5], ]
        self.oih_infos.append(self.cc.add_cdobj(objcm, rel_pos, rel_rotmat, into_list))
        return rel_pos, rel_rotmat

    def release(self, hnd_name, objcm, jawwidth=None):
        """
        the objcm is added as a part of the robot_s to the cd checker
        :param jawwidth:
        :param objcm:
        :return:
        """
        if hnd_name not in self.hnd_dict:
            raise ValueError("Hand name does not exist!")
        if jawwidth is not None:
            self.hnd_dict[hnd_name].jaw_to(jawwidth)
        for obj_info in self.oih_infos:
            if obj_info['collision_model'] is objcm:
                self.cc.delete_cdobj(obj_info)
                self.oih_infos.remove(obj_info)
                break

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
        self.body.gen_mesh_model(tcp_loc_pos=None,
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

    base = wd.World(cam_pos=[2, 0, 1.5], lookat_pos=[0, 0, .2])
    gm.gen_frame().attach_to(base)
    xarm = XArmLite6WRSGripper(enable_cc=True)
    # rand_conf = xarm.rand_conf(component_name='arm')
    # xarm.fk("arm", rand_conf)
    xarm.gen_meshmodel().attach_to(base)
    # xarm.fk("arm", np.array([0, 0, 0, 0, 0, -np.pi / 2]))
    print("Is self collided?", xarm.is_collided())
    base.run()
