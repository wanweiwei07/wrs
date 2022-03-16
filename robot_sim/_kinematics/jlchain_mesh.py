import numpy as np
import modeling.geometric_model as gm
import modeling.collision_model as cm
import modeling.model_collection as mc
import basis.robot_math as rm


class JLChainMesh(object):
    """
    The mesh generator class for JntLnks
    NOTE: it is unnecessary to attach a nodepath to render repeatedly
    once attached, it is always there. update the joint angles
    will change the attached model directly
    """

    def __init__(self, jlobject, cdprimitive_type='box', cdmesh_type='triangles'):
        """
        author: weiwei
        date: 20200331
        """
        self.jlobject = jlobject
        for id in range(self.jlobject.ndof + 1):
            if self.jlobject.lnks[id]['mesh_file'] is not None and self.jlobject.lnks[id]['collision_model'] is None:
                # in case the collision model is directly set, it allows manually specifying cd primitives
                # instead of auto initialization. Steps: 1. keep meshmodel to None; 2. directly set cm
                self.jlobject.lnks[id]['collision_model'] = cm.CollisionModel(self.jlobject.lnks[id]['mesh_file'],
                                                                             cdprimit_type=cdprimitive_type,
                                                                             cdmesh_type=cdmesh_type)
                self.jlobject.lnks[id]['collision_model'].set_scale(self.jlobject.lnks[id]['scale'])

    def gen_meshmodel(self,
                      tcp_jnt_id=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcpcs=True,
                      toggle_jntscs=False,
                      name='robot_mesh',
                      rgba=None):
        mm_collection = mc.ModelCollection(name=name)
        for id in range(self.jlobject.ndof + 1):
            if self.jlobject.lnks[id]['collision_model'] is not None:
                this_collisionmodel = self.jlobject.lnks[id]['collision_model'].copy()
                pos = self.jlobject.lnks[id]['gl_pos']
                rotmat = self.jlobject.lnks[id]['gl_rotmat']
                this_collisionmodel.set_homomat(rm.homomat_from_posrot(pos, rotmat))
                this_rgba = self.jlobject.lnks[id]['rgba'] if rgba is None else rgba
                this_collisionmodel.set_rgba(this_rgba)
                this_collisionmodel.attach_to(mm_collection)
        # tool center coord
        if toggle_tcpcs:
            self._toggle_tcpcs(mm_collection,
                               tcp_jnt_id,
                               tcp_loc_pos,
                               tcp_loc_rotmat,
                               tcpic_rgba=np.array([.5, 0, 1, 1]), tcpic_thickness=.0062)
        # toggle all coord
        if toggle_jntscs:
            alpha = 1 if rgba == None else rgba[3]
            self._toggle_jntcs(mm_collection,
                               jntcs_thickness=.0062,
                               alpha=alpha)
        return mm_collection

    def gen_stickmodel(self,
                       rgba=np.array([.5, 0, 0, 1]),
                       thickness=.01,
                       joint_ratio=1.62,
                       link_ratio=.62,
                       tcp_jnt_id=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcpcs=True,
                       toggle_jntscs=False,
                       toggle_connjnt=False,
                       name='robot_stick'):
        """
        generate the stick model for a jntlnk object
        snp means stick nodepath
        :param rgba:
        :param tcp_jnt_id:
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param toggle_tcpcs:
        :param toggle_jntscs:
        :param toggle_connjnt: draw the connecting joint explicitly or not
        :param name:
        :return:

        author: weiwei
        date: 20200331, 20201006
        """
        stickmodel = mc.ModelCollection(name=name)
        id = 0
        loopdof = self.jlobject.ndof + 1
        if toggle_connjnt:
            loopdof = self.jlobject.ndof + 2
        while id < loopdof:
            cjid = self.jlobject.jnts[id]['child']
            jgpos = self.jlobject.jnts[id]['gl_posq']  # joint global pos
            cjgpos = self.jlobject.jnts[cjid]['gl_pos0']  # child joint global pos
            jgmtnax = self.jlobject.jnts[id]["gl_motionax"]  # joint global rot ax
            gm.gen_stick(spos=jgpos, epos=cjgpos, thickness=thickness, type="rect", rgba=rgba).attach_to(stickmodel)
            if id > 0:
                if self.jlobject.jnts[id]['type'] == "revolute":
                    gm.gen_stick(spos=jgpos - jgmtnax * thickness, epos=jgpos + jgmtnax * thickness, type="rect",
                                 thickness=thickness * joint_ratio, rgba=np.array([.3, .3, .2, rgba[3]])).attach_to(stickmodel)
                if self.jlobject.jnts[id]['type'] == "prismatic":
                    jgpos0 = self.jlobject.jnts[id]['gl_pos0']
                    gm.gen_stick(spos=jgpos0, epos=jgpos, type="round", thickness=thickness * joint_ratio,
                                 rgba=np.array([.2, .3, .3, rgba[3]])).attach_to(stickmodel)
            id = cjid
        # tool center coord
        if toggle_tcpcs:
            self._toggle_tcpcs(stickmodel, tcp_jnt_id, tcp_loc_pos, tcp_loc_rotmat,
                               tcpic_rgba=rgba + np.array([0, 0, 1, 0]), tcpic_thickness=thickness * link_ratio)
        # toggle all coord
        if toggle_jntscs:
            self._toggle_jntcs(stickmodel, jntcs_thickness=thickness * link_ratio, alpha=rgba[3])
        return stickmodel

    def gen_endsphere(self, rgba=None, name=''):
        """
        generate an end sphere (es) to show the trajectory of the end effector

        :param jlobject: a JntLnk object
        :param rbga: color of the arm
        :return: null

        author: weiwei
        date: 20181003madrid, 20200331
        """
        pass
        # eesphere = gm.StaticGeometricModel(name=name)
        # if rgba is not None:
        #     gm.gen_sphere(pos=self.jlobject.jnts[-1]['linkend'], radius=.025, rgba=rgba).attach_to(eesphere)
        # return gm.StaticGeometricModel(eesphere)

    def _toggle_tcpcs(self,
                      parent_model,
                      tcp_jnt_id,
                      tcp_loc_pos,
                      tcp_loc_rotmat,
                      tcpic_rgba,
                      tcpic_thickness,
                      tcpcs_thickness=None,
                      tcpcs_length=None):
        """
        :param parent_model: where to draw the frames to
        :param tcp_jnt_id: single id or a list of ids
        :param tcp_loc_pos:
        :param tcp_loc_rotmat:
        :param tcpic_rgba: color that used to render the tcp indicator
        :param tcpic_thickness: thickness the tcp indicator
        :param tcpcs_thickness: thickness the tcp coordinate frame
        :return:

        author: weiwei
        date: 20201125
        """
        if tcp_jnt_id is None:
            tcp_jnt_id = self.jlobject.tcp_jnt_id
        if tcp_loc_pos is None:
            tcp_loc_pos = self.jlobject.tcp_loc_pos
        if tcp_loc_rotmat is None:
            tcp_loc_rotmat = self.jlobject.tcp_loc_rotmat
        if tcpcs_thickness is None:
            tcpcs_thickness = tcpic_thickness
        if tcpcs_length is None:
            tcpcs_length = tcpcs_thickness * 15
        tcp_gl_pos, tcp_gl_rotmat = self.jlobject.get_gl_tcp(tcp_jnt_id,
                                                             tcp_loc_pos,
                                                             tcp_loc_rotmat)
        if isinstance(tcp_gl_pos, list):
            for i, jid in enumerate(tcp_jnt_id):
                jgpos = self.jlobject.jnts[jid]['gl_posq']
                gm.gen_dashstick(spos=jgpos,
                                 epos=tcp_gl_pos[i],
                                 thickness=tcpic_thickness,
                                 rgba=tcpic_rgba,
                                 type="round").attach_to(parent_model)
                gm.gen_mycframe(pos=tcp_gl_pos[i],
                                rotmat=tcp_gl_rotmat[i],
                                length=tcpcs_length,
                                thickness=tcpcs_thickness,
                                alpha=tcpic_rgba[3]).attach_to(parent_model)
        else:
            jgpos = self.jlobject.jnts[tcp_jnt_id]['gl_posq']
            gm.gen_dashstick(spos=jgpos,
                             epos=tcp_gl_pos,
                             thickness=tcpic_thickness,
                             rgba=tcpic_rgba,
                             type="round").attach_to(parent_model)
            gm.gen_mycframe(pos=tcp_gl_pos,
                            rotmat=tcp_gl_rotmat,
                            length=tcpcs_length,
                            thickness=tcpcs_thickness,
                            alpha=tcpic_rgba[3]).attach_to(parent_model)

    def _toggle_jntcs(self, parentmodel, jntcs_thickness, jntcs_length=None, alpha=1):
        """
        :param parentmodel: where to draw the frames to
        :return:

        author: weiwei
        date: 20201125
        """
        if jntcs_length is None:
            jntcs_length = jntcs_thickness * 15
        for id in self.jlobject.tgtjnts:
            gm.gen_dashframe(pos=self.jlobject.jnts[id]['gl_pos0'],
                             rotmat=self.jlobject.jnts[id]['gl_rotmat0'],
                             length=jntcs_length,
                             thickness=jntcs_thickness,
                             alpha=alpha).attach_to(parentmodel)
            gm.gen_frame(pos=self.jlobject.jnts[id]['gl_posq'],
                         rotmat=self.jlobject.jnts[id]['gl_rotmatq'],
                         length=jntcs_length,
                         thickness=jntcs_thickness,
                         alpha=alpha).attach_to(parentmodel)
