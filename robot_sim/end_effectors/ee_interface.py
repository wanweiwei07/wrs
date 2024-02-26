import copy
import numpy as np
import modeling.model_collection as mmc
import modeling.collision_model as mcm
import robot_sim._kinematics.jl as jl
import robot_sim._kinematics.jlchain as rkjl
import robot_sim._kinematics.collision_checker as rkcc
import modeling.geometric_model as mgm
import modeling.constant as mc
import basis.robot_math as rm


class EEInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type=mc.CDMType.AABB, name="end_effector"):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cdmesh_type = cdmesh_type  # aabb, convexhull, or triangles
        # joints
        # -- coupling --
        # no coupling by default, change the pos if the coupling existed
        self.coupling = rkjl.JLChain(pos=self.pos, rotmat=self.rotmat, n_dof=0, name=name+"_coupling")
        self.coupling.tcp_loc_pos = np.array([0, 0, 0])
        self.coupling.anchor.name = "coupling_anchor"
        # toggle on the following part to assign an explicit mesh model to a coupling
        # self.coupling.jnts[0].link = rkjl.create_link(mesh_file=os.path.join(this_dir, "meshes", "xxx.stl"))
        # self.coupling.jnts[0].link = mcm.gen_stick(spos=self.coupling.anchor.pos, epos = self.coupling.jnts[0].pos)
        # self.coupling.jnts[0].lnks.rgba = [.2, .2, .2, 1]
        self.coupling.finalize(ik_solver=None)
        # action center, acting point of the tool
        self.action_center_pos = np.zeros(3)
        self.action_center_rotmat = np.eye(3)
        # collision detection
        self.cc = None
        # cd mesh collection for precise collision checking
        self.cdmesh_collection = mmc.ModelCollection()
        # object grasped/held/attached to end-effector; oiee = object in end-effector
        self.oiee_list = []

    def update_oiee(self):
        """
        :return:
        author: weiwei
        date: 20230807
        """
        for oiee in self.oiee_list:
            gl_pos, gl_rotmat = self.cvt_loc_tcp_to_gl(oiee.loc_pos, )
            oiee.update_globals()

    def hold(self, obj_cmodel, **kwargs):
        """
        the objcm is saved into an oiee_list, while considering its relative pose to the ee's pos and rotmat
        **kwargs is for polyphorism purpose
        :param obj_cmodel: a collision model
        :return:
        author: weiwei
        date: 20230811
        """
        obj_pos = obj_cmodel.pos
        obj_rotmat = obj_cmodel.rotmat
        rel_pos, rel_rotmat = rm.rel_pose(obj_pos, obj_rotmat, self.pos, self.rotmat)
        self.oiee_list.append(jl.Link(loc_pos = rel_pos, loc_rotmat=rel_rotmat, cmodel=obj_cmodel))

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return_val = self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        return return_val

    def is_mesh_collided(self, cmodel_list=[], toggle_debug=False):
        for i, cdme in enumerate(self.cdmesh_elements):
            if cdme.cmodel is not None:
                is_collided, collision_points = cdme.cmodel.is_mcdwith(cmodel_list, True)
                if is_collided:
                    if toggle_debug:
                        cdme.show_cdmesh()
                        for cmodel in cmodel_list:
                            cmodel.show_cdmesh()
                        for point in collision_points:
                            import modeling.geometric_model as mgm
                            mgm.gen_sphere(point, radius=.001).attach_to(base)
                        print("mesh collided")
                    return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        for i, cdme in enumerate(self.cdmesh_elements):
            cdme.cmodel.show_cdmesh()

        # for i, cdelement in enumerate(self.cc.cce_dict):
        #     pos = cdelement['gl_pos']
        #     rotmat = cdelement['gl_rotmat']
        #     self.cdmesh_collection.cm_list[i].set_pos(pos)
        #     self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        # self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='ee_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='ee_meshmodel'):
        raise NotImplementedError

    def _toggle_tcpcs(self, parent):
        action_center_gl_pos = self.rotmat.dot(self.action_center_pos) + self.pos
        action_center_gl_rotmat = self.rotmat.dot(self.action_center_rotmat)
        mgm.gen_dashed_stick(spos=self.pos,
                             epos=action_center_gl_pos,
                             radius=.0062,
                             rgba=[.5, 0, 1, 1],
                             type="round").attach_to(parent)
        mgm.gen_myc_frame(pos=action_center_gl_pos, rotmat=action_center_gl_rotmat).attach_to(parent)

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and pdndp
        :return:
        """
        for cdelement in self.cc.cce_dict:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.cd_trav.addCollider(child, self_copy.cc.cd_handler)
        return self_copy
