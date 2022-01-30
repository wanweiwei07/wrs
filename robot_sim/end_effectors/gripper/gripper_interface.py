import copy
import numpy as np
import modeling.model_collection as mc
import basis.robot_math as rm
import robot_sim._kinematics.jlchain as jl
import robot_sim._kinematics.collision_checker as cc


class GripperInterface(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), cdmesh_type='aabb', name='gripper'):
        self.name = name
        self.pos = pos
        self.rotmat = rotmat
        self.cdmesh_type = cdmesh_type # aabb, convexhull, or triangles
        # joints
        # - coupling - No coupling by default
        self.coupling = jl.JLChain(pos=self.pos, rotmat=self.rotmat, homeconf=np.zeros(0), name='coupling')
        self.coupling.jnts[1]['loc_pos'] = np.array([0, 0, .0])
        self.coupling.lnks[0]['name'] = 'coupling_lnk0'
        # toggle on the following part to assign an explicit mesh model to a coupling
        # self.coupling.lnks[0]['meshfile'] = os.path.join(this_dir, "meshes", "xxx.stl")
        # self.coupling.lnks[0]['rgba'] = [.2, .2, .2, 1]
        self.coupling.reinitialize()
        # jaw center
        self.jaw_center_pos = np.zeros(3)
        self.jaw_center_rotmat = np.eye(3)
        # jaw width
        self.jawwidth_rng = [0.0, 5.0]
        # collision detection
        self.cc = None
        # cd mesh collection for precise collision checking
        self.cdmesh_collection = mc.ModelCollection()

    def is_collided(self, obstacle_list=[], otherrobot_list=[]):
        """
        Interface for "is cdprimit collided", must be implemented in child class
        :param obstacle_list:
        :param otherrobot_list:
        :return:
        author: weiwei
        date: 20201223
        """
        return_val =  self.cc.is_collided(obstacle_list=obstacle_list, otherrobot_list=otherrobot_list)
        return return_val

    def is_mesh_collided(self, objcm_list=[], toggle_debug=False):
        for i, cdelement in enumerate(self.all_cdelements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
            iscollided, collided_points = self.cdmesh_collection.cm_list[i].is_mcdwith(objcm_list, True)
            if iscollided:
                if toggle_debug:
                    print(self.cdmesh_collection.cm_list[i].get_homomat())
                    self.cdmesh_collection.cm_list[i].show_cdmesh()
                    for objcm in objcm_list:
                        objcm.show_cdmesh()
                    for point in collided_points:
                        import modeling.geometric_model as gm
                        gm.gen_sphere(point, radius=.001).attach_to(base)
                    print("collided")
                return True
        return False

    def fix_to(self, pos, rotmat):
        raise NotImplementedError

    def fk(self, motion_val):
        raise NotImplementedError

    def jaw_to(self, jaw_width):
        raise NotImplementedError

    def get_jawwidth(self):
        raise NotImplementedError

    def grip_at_with_jczy(self, gl_jaw_center_pos, gl_jaw_center_z, gl_jaw_center_y, jaw_width):
        """
        :param gl_jaw_center_pos:
        :param gl_jaw_center_z: jaw_center's approaching direction
        :param gl_jaw_center_y: jaw_center's opening direction
        :param jaw_width:
        :return:
        """
        gl_jaw_center_rotmat = np.eye(3)
        gl_jaw_center_rotmat[:, 2] = rm.unit_vector(gl_jaw_center_z)
        gl_jaw_center_rotmat[:, 1] = rm.unit_vector(gl_jaw_center_y)
        gl_jaw_center_rotmat[:, 0] = np.cross(gl_jaw_center_rotmat[:3, 1], gl_jaw_center_rotmat[:3, 2])
        return self.grip_at_with_jcpose(gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width)

    def grip_at_with_jcpose(self, gl_jaw_center_pos, gl_jaw_center_rotmat, jaw_width):
        """
        :param gl_jaw_center_pos:
        :param gl_jaw_center_rotmat: jaw_center's rotmat
        :param jaw_width:
        :return:
        """
        self.jaw_to(jaw_width)
        eef_root_rotmat = gl_jaw_center_rotmat.dot(self.jaw_center_rotmat.T)
        eef_root_pos = gl_jaw_center_pos - eef_root_rotmat.dot(self.jaw_center_pos)
        self.fix_to(eef_root_pos, eef_root_rotmat)
        return [jaw_width, gl_jaw_center_pos, gl_jaw_center_rotmat, eef_root_pos, eef_root_rotmat]

    def show_cdprimit(self):
        self.cc.show_cdprimit()

    def unshow_cdprimit(self):
        self.cc.unshow_cdprimit()

    def show_cdmesh(self):
        for i, cdelement in enumerate(self.cc.all_cdelements):
            pos = cdelement['gl_pos']
            rotmat = cdelement['gl_rotmat']
            self.cdmesh_collection.cm_list[i].set_pos(pos)
            self.cdmesh_collection.cm_list[i].set_rotmat(rotmat)
        self.cdmesh_collection.show_cdmesh()

    def unshow_cdmesh(self):
        self.cdmesh_collection.unshow_cdmesh()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=False,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='gripper_stickmodel'):
        raise NotImplementedError

    def gen_meshmodel(self,
                      toggle_tcpcs=False,
                      toggle_jntscs=False,
                      rgba=None,
                      name='gripper_meshmodel'):
        raise NotImplementedError

    def enable_cc(self):
        self.cc = cc.CollisionChecker("collision_checker")

    def disable_cc(self):
        """
        clear pairs and nodepath
        :return:
        """
        for cdelement in self.cc.all_cdelements:
            cdelement['cdprimit_childid'] = -1
        self.cc = None

    def copy(self):
        self_copy = copy.deepcopy(self)
        # deepcopying colliders are problematic, I have to update it manually
        if self.cc is not None:
            for child in self_copy.cc.np.getChildren():
                self_copy.cc.ctrav.addCollider(child, self_copy.cc.chan)
        return self_copy

