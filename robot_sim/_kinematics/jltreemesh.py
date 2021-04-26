import os
import numpy as np
import copy
from panda3d.core import NodePath
import modeling.geometric_model as gm
import modeling.collision_model as cm
import basis.robot_math as rm


class JLTreeMesh(object):
    """
    The mesh generator class for JLTree

    NOTE: it is unnecessary to attach a nodepath to render repeatedly
    once attached, it is always there. update the joint angles
    will change the attached model directly
    """

    def __init__(self, jltree_obj):
        """
        author: weiwei
        date: 20200331
        """
        self.jltree_obj = jltree_obj

    def gen_stickmodel(self, rgba=np.array([.5, 0, 0, 1]), thickness=.01, jointratio=1.62, linkratio=.62,
                       tcp_jntid=None, tcp_localpos=None, tcp_localrotmat=None,
                       toggletcpcs=True, togglejntscs=False, togglecntjnt=False, name='robotstick'):
        """
        generate a stick model for self.jltree_obj
        :param rgba:
        :param tcp_jntid:
        :param tcp_localpos:
        :param tcp_localrotmat:
        :param toggletcpcs:
        :param togglejntscs:
        :param togglecntjnt: draw the connecting joint explicitly or not
        :param name:
        :return:
        author: weiwei
        date: 20200331, 20201006, 20201205
        """
        stickmodel = gm.StaticGeometricModel(name=name)
        id = 0
        loopdof = self.jlobject.ndof + 1
        if togglecntjnt:
            loopdof = self.jlobject.ndof + 2
        while id < loopdof:
            cjid = self.jlobject.joints[id]['child']
            jgpos = self.jlobject.joints[id]['g_posq']  # joint global pos
            cjgpos = self.jlobject.joints[cjid]['g_pos0']  # child joint global pos
            jgmtnax = self.jlobject.joints[id]["g_mtnax"]  # joint global rot ax
            gm.gen_stick(spos=jgpos, epos=cjgpos, thickness=thickness, type="rect", rgba=rgba).attach_to(stickmodel)
            if id > 0:
                if self.jlobject.joints[id]['type'] == "revolute":
                    gm.gen_stick(spos=jgpos - jgmtnax * thickness, epos=jgpos + jgmtnax * thickness, type="rect",
                                 thickness=thickness * jointratio, rgba=np.array([.3, .3, .2, 1])).attach_to(stickmodel)
                if self.jlobject.joints[id]['type'] == "prismatic":
                    jgpos0 = self.jlobject.joints[id]['g_pos0']
                    gm.gen_stick(spos=jgpos0, epos=jgpos, type="round", hickness=thickness * jointratio,
                                 rgba=np.array([.2, .3, .3, 1])).attach_to(stickmodel)
            id = cjid
        # tool center coord
        if toggletcpcs:
            self._toggle_tcpcs(stickmodel, tcp_jntid, tcp_localpos, tcp_localrotmat,
                               tcpic_rgba=rgba + np.array([0, 0, 1, 0]), tcpic_thickness=thickness * linkratio)
        # toggle all coord
        if togglejntscs:
            self._toggle_jntcs(stickmodel, jntcs_thickness=thickness * linkratio)
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

        eesphere = gm.StaticGeometricModel(name=name)
        if rgba is not None:
            gm.gen_sphere(pos=self.jlobject.joints[-1]['linkend'], radius=.025, rgba=rgba).attach_to(eesphere)
        return gm.StaticGeometricModel(eesphere)

    def _toggle_tcpcs(self, parentmodel, tcp_jntid, tcp_localpos, tcp_localrotmat, tcpic_rgba, tcpic_thickness,
                      tcpcs_thickness=None, tcpcs_length=None):
        """
        :param parentmodel: where to draw the frames to
        :param tcp_jntid: single id or a list of ids
        :param tcp_localpos:
        :param tcp_localrotmat:
        :param tcpic_rgba: color that used to render the tcp indicator
        :param tcpic_thickness: thickness the tcp indicator
        :param tcpcs_thickness: thickness the tcp coordinate frame
        :return:

        author: weiwei
        date: 20201125
        """
        if tcp_jntid is None:
            tcp_jntid = self.jlobject.tcp_jntid
        if tcp_localpos is None:
            tcp_localpos = self.jlobject.tcp_localpos
        if tcp_localrotmat is None:
            tcp_localrotmat = self.jlobject.tcp_localrotmat
        if tcpcs_thickness is None:
            tcpcs_thickness = tcpic_thickness
        if tcpcs_length is None:
            tcpcs_length = tcpcs_thickness * 15
        tcp_globalpos, tcp_globalrotmat = self.jlobject.get_gl_tcp(tcp_jntid, tcp_localpos, tcp_localrotmat)
        if isinstance(tcp_globalpos, list):
            for i, jid in enumerate(tcp_jntid):
                jgpos = self.jlobject.joints[jid]['g_posq']
                gm.gen_dumbbell(spos=jgpos, epos=tcp_globalpos[i], thickness=tcpic_thickness,
                                rgba=tcpic_rgba).attach_to(parentmodel)
                gm.gen_frame(pos=tcp_globalpos[i], rotmat=tcp_globalrotmat[i], length=tcpcs_length,
                             thickness=tcpcs_thickness, alpha=1).attach_to(parentmodel)
        else:
            jgpos = self.jlobject.joints[tcp_jntid]['g_posq']
            gm.gen_dumbbell(spos=jgpos, epos=tcp_globalpos, thickness=tcpic_thickness, rgba=tcpic_rgba).attach_to(
                parentmodel)
            gm.gen_frame(pos=tcp_globalpos, rotmat=tcp_globalrotmat, length=tcpcs_length, thickness=tcpcs_thickness,
                        alpha=1).attach_to(parentmodel)

    def _toggle_jntcs(self, parentmodel, jntcs_thickness, jntcs_length=None):
        """
        :param parentmodel: where to draw the frames to
        :return:

        author: weiwei
        date: 20201125
        """
        if jntcs_length is None:
            jntcs_length = jntcs_thickness * 15
        for id in self.jlobject.tgtjnts:
            gm.gen_dashframe(pos=self.jlobject.joints[id]['g_pos0'], rotmat=self.jlobject.joints[id]['g_rotmat0'],
                            length=jntcs_length, thickness=jntcs_thickness).attach_to(parentmodel)
            gm.gen_frame(pos=self.jlobject.joints[id]['g_posq'], rotmat=self.jlobject.joints[id]['g_rotmatq'],
                        length=jntcs_length, thickness=jntcs_thickness, alpha=1).attach_to(parentmodel)
