import robot_sim._kinematics.jlchain as rkjlc
import numpy as np
import networkx as nx
from robot_sim.urdf.urdf_parser import URDF


class URDFRobot(object):

    def __init__(self, urdf_file="robot.urdf"):
        self._urdf_file = urdf_file
        self._urdf = URDF.load(urdf_file)
        self._jlg_segments = self.urdf.segment(toggle_debug=True)
        self.components = []
        for jlg in self._jlg_segments:
            if jlg.number_of_edges() == 1:
                # anchor
                pass
            else:
                sorted_nodes = list(nx.topological_sort(jlg))
                sorted_edges = list(zip(sorted_nodes[:-1], sorted_nodes[1:]))
                anchor = jlg[sorted_edges[0][0]][sorted_edges[0][1]]['joint']
                flange = jlg[sorted_edges[-1][0]][sorted_edges[-1][1]]['joint']
                jlc = rkjlc.JLChain(pos=anchor.origin[:3, 3], rotmat=anchor.origin[:3, :3], name=jlg.graph['name'],
                                    n_dof=jlg.number_of_edges())
                for i, jnt in enumerate(jlc.jnts):
                    urdf_jnt = jlg[sorted_edges[i][0]][sorted_edges[i][1]]['joint']
                    if urdf_jnt.joint_type == 'prismatic':
                        jnt.change_type(rkjlc.rkc.JntType.PRISMATIC,
                                        motion_range=np.asarray([urdf_jnt.limit.lower, urdf_jnt.limit.upper]))
                    jnt.loc_pos = [i]['origin'][:3, 3]
                    jnt.loc_motion_ax = jlg[jnt.name]['axis'][:3]
                    jnt.motion_range = jlg[jnt.name]['motion_range']
