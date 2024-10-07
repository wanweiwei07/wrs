import os
import math
import numpy as np
import wrs.modeling.model_collection as mc


class TBM(object):

    def __init__(self, pos=np.zeros(3), rotmat=np.eye(3), name="tbm"):
        self.pos = pos
        self.rotmat = rotmat
        self.name = name
        this_dir, this_filename = os.path.split(__file__)
        # head
        self.head = jl.JLChain(pos=pos, rotmat=rotmat, home_conf=np.zeros(1), name='head')
        self.head.jnts[1]['loc_pos'] = np.zeros(3)
        self.head.jnts[1]['loc_motionax'] = np.array([1, 0, 0])
        self.head.lnks[0]['name'] = 'tbm_front_shield'
        self.head.lnks[0]['loc_pos'] = np.array([0, 0, 0])
        self.head.lnks[0]['mesh_file'] = os.path.join(this_dir, 'meshes', 'tbm_front_shield.stl')
        self.head.lnks[0]['rgba'] = [.7, .7, .7, .3]
        self.head.lnks[1]['name'] = 'tbm_cutter_head'
        self.head.lnks[1]['loc_pos'] = np.array([0, 0, 0])
        self.head.lnks[1]['mesh_file'] = os.path.join(this_dir, 'meshes', 'tbm_cutter_head.stl')
        self.head.lnks[1]['rgba'] = [.7, .3, .3, 1]
        self.head.tgtjnts = [1]
        self.head.finalize()
        # cutter
        self.cutters = []
        self.cutter_pos_dict = {'0': [np.array([0.410, 0.0, 0.341 + 0.175]),
                                      np.array([0.410, 0.0, 0.590 + 0.175]),
                                      np.array([0.410, 0.0, 0.996 + 0.175]),
                                      np.array([0.410, 0.0, 1.245 + 0.175]),
                                      np.array([0.410, 0.0, 2.162 + 0.175]),
                                      np.array([0.410, 0.0, 2.802 + 0.175]),
                                      np.array([0.410, 0.0, 3.442 + 0.175]),
                                      np.array([0.410, 0.0, 4.082 + 0.175]),
                                      np.array([0.410, 0.0, 4.662 + 0.175]),
                                      np.array([0.410, 0.0, 5.712 + 0.175])],
                                '6': [np.array([0.410, 0.0, -0.341 - 0.175]),
                                      np.array([0.410, 0.0, -0.590 - 0.175]),
                                      np.array([0.410, 0.0, -0.996 - 0.175]),
                                      np.array([0.410, 0.0, -1.245 - 0.175]),
                                      np.array([0.410, 0.0, -2.020 - 0.175]),
                                      np.array([0.410, 0.0, -2.660 - 0.175]),
                                      np.array([0.410, 0.0, -3.300 - 0.175]),
                                      np.array([0.410, 0.0, -3.915 - 0.175]),
                                      np.array([0.410, 0.0, -4.515 - 0.175]),
                                      np.array([0.410, 0.0, -5.560 - 0.175])],
                                '3': [np.array([0.410, -0.050 + 0.175, 0.0]),
                                      np.array([0.410, 0.177 + 0.175, 0.0]),
                                      np.array([0.410, 0.605 + 0.175, 0.0]),
                                      np.array([0.410, 0.854 + 0.175, 0.0]),
                                      np.array([0.410, 1.450 + 0.175, 0.0]),
                                      np.array([0.410, 1.699 + 0.175, 0.0]),
                                      np.array([0.410, 2.562 + 0.175, 0.0]),
                                      np.array([0.410, 3.202 + 0.175, 0.0]),
                                      np.array([0.410, 3.837 + 0.175, 0.0]),
                                      np.array([0.410, 4.437 + 0.175, 0.0]),
                                      np.array([0.410, 5.037 + 0.175, 0.0]),
                                      np.array([0.410, 5.862 + 0.175, 0.0]),
                                      np.array([0.410, 6.912 + 0.175, 0.0])],
                                '9': [np.array([0.410, 0.050 - 0.175, 0.0]),
                                      np.array([0.410, -0.177 - 0.175, 0.0]),
                                      np.array([0.410, -0.605 - 0.175, 0.0]),
                                      np.array([0.410, -0.854 - 0.175, 0.0]),
                                      np.array([0.410, -1.450 - 0.175, 0.0]),
                                      np.array([0.410, -1.699 - 0.175, 0.0]),
                                      np.array([0.410, -2.260 - 0.175, 0.0]),
                                      np.array([0.410, -2.900 - 0.175, 0.0]),
                                      np.array([0.410, -3.540 - 0.175, 0.0]),
                                      np.array([0.410, -4.140 - 0.175, 0.0]),
                                      np.array([0.410, -4.740 - 0.175, 0.0]),
                                      np.array([0.410, -5.715 - 0.175, 0.0]),
                                      np.array([0.410, -6.765 - 0.175, 0.0]),
                                      np.array([0.417 - 0.175 * math.sin(math.pi / 6),
                                                -7.637 - 0.175 * math.cos(math.pi / 6), 0.0])],
                                '1.5': [np.array([0.410, 1.259 + 0.175 * math.sin(math.pi / 4),
                                                  1.259 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 1.711 + 0.175 * math.sin(math.pi / 4),
                                                  1.711 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 2.164 + 0.175 * math.sin(math.pi / 4),
                                                  2.164 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 2.609 + 0.175 * math.sin(math.pi / 4),
                                                  2.609 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 3.033 + 0.175 * math.sin(math.pi / 4),
                                                  3.033 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 3.882 + 0.175 * math.sin(math.pi / 4),
                                                  3.882 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 4.731 + 0.175 * math.sin(math.pi / 4),
                                                  4.731 + 0.175 * math.cos(math.pi / 4)]),
                                        np.array([0.410, 5.101 + 0.175 * math.sin(math.pi / 4),
                                                  5.101 + 0.175 * math.cos(math.pi / 4)])],
                                '4.5': [np.array([0.410, 1.541 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -1.541 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 1.994 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -1.994 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 2.447 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -2.447 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 2.874 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -2.874 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 3.299 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -3.299 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 4.253 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -4.253 + 0.175 * math.cos(3 * math.pi / 4)]),
                                        np.array([0.410, 4.996 + 0.175 * math.sin(3 * math.pi / 4),
                                                  -4.996 + 0.175 * math.cos(3 * math.pi / 4)])],
                                '7.5': [np.array([0.410, -1.315 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -1.315 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -1.768 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -1.768 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -2.220 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -2.220 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -2.662 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -2.662 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -3.086 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -3.086 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -4.094 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -4.094 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.410, -4.837 + 0.175 * math.sin(-3 * math.pi / 4),
                                                  -4.837 + 0.175 * math.cos(-3 * math.pi / 4)]),
                                        np.array([0.436 + 0.175 * math.sin(-math.pi / 4),
                                                  -5.429 + 0.175 * math.sin(-3 * math.pi / 4) * math.sin(
                                                      3 * math.pi / 4),
                                                  -5.429 + 0.175 * math.cos(-3 * math.pi / 4) * math.sin(
                                                      3 * math.pi / 4)])],
                                '10.5': [np.array([0.410, -1.485 + 0.175 * math.sin(-math.pi / 4),
                                                   1.485 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -1.937 + 0.175 * math.sin(-math.pi / 4),
                                                   1.937 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -2.390 + 0.175 * math.sin(-math.pi / 4),
                                                   2.390 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -2.821 + 0.175 * math.sin(-math.pi / 4),
                                                   2.821 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -3.246 + 0.175 * math.sin(-math.pi / 4),
                                                   3.246 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -3.723 + 0.175 * math.sin(-math.pi / 4),
                                                   3.723 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -4.200 + 0.175 * math.sin(-math.pi / 4),
                                                   4.200 + 0.175 * math.cos(-math.pi / 4)]),
                                         np.array([0.410, -4.943 + 0.175 * math.sin(-math.pi / 4),
                                                   4.943 + 0.175 * math.cos(-math.pi / 4)])],
                                '0.75': [np.array([0.410, 2.072 + 0.175 * math.sin(math.pi / 8),
                                                   5.003 + 0.175 * math.cos(math.pi / 8)]),
                                         np.array([0.410, 2.503 + 0.175 * math.sin(math.pi / 8),
                                                   6.042 + 0.175 * math.cos(math.pi / 8)]),
                                         np.array([0.410, 2.818 + 0.175 * math.sin(math.pi / 8),
                                                   6.804 + 0.175 * math.cos(math.pi / 8)])],
                                '2.25': [np.array([0.410, 4.656 + 0.175 * math.sin(3 * math.pi / 8),
                                                   1.929 + 0.175 * math.cos(3 * math.pi / 8)]),
                                         np.array([0.410, 5.626 + 0.175 * math.sin(3 * math.pi / 8),
                                                   2.331 + 0.175 * math.cos(3 * math.pi / 8)]),
                                         np.array([0.410, 6.596 + 0.175 * math.sin(3 * math.pi / 8),
                                                   2.732 + 0.175 * math.cos(3 * math.pi / 8)])],
                                '3.75': [np.array([0.410, 4.795 + 0.175 * math.sin(5 * math.pi / 8),
                                                   -1.986 + 0.175 * math.cos(5 * math.pi / 8)]),
                                         np.array([0.410, 5.973 + 0.175 * math.sin(5 * math.pi / 8),
                                                   -2.474 + 0.175 * math.cos(5 * math.pi / 8)]),
                                         np.array([0.410, 6.735 + 0.175 * math.sin(5 * math.pi / 8),
                                                   -2.790 + 0.175 * math.cos(5 * math.pi / 8)])],
                                '5.25': [np.array([0.410, 1.900 + 0.175 * math.sin(7 * math.pi / 8),
                                                   -4.587 + 0.175 * math.cos(7 * math.pi / 8)]),
                                         np.array([0.410, 2.388 + 0.175 * math.sin(7 * math.pi / 8),
                                                   -5.765 + 0.175 * math.cos(7 * math.pi / 8)]),
                                         np.array([0.410, 2.761 + 0.175 * math.sin(7 * math.pi / 8),
                                                   -6.666 + 0.175 * math.cos(7 * math.pi / 8)])],
                                '6.75': [np.array([0.410, -1.986 + 0.175 * math.sin(-7 * math.pi / 8),
                                                   -4.795 + 0.175 * math.cos(-7 * math.pi / 8)]),
                                         np.array([0.410, -2.445 + 0.175 * math.sin(-7 * math.pi / 8),
                                                   -5.904 + 0.175 * math.cos(-7 * math.pi / 8)]),
                                         np.array([0.410, -2.790 + 0.175 * math.sin(-7 * math.pi / 8),
                                                   -6.735 + 0.175 * math.cos(-7 * math.pi / 8)])],
                                '8.25': [np.array([0.410, -4.726 + 0.175 * math.sin(-5 * math.pi / 8),
                                                   -1.957 + 0.175 * math.cos(-5 * math.pi / 8)]),
                                         np.array([0.410, -5.696 + 0.175 * math.sin(-5 * math.pi / 8),
                                                   -2.359 + 0.175 * math.cos(-5 * math.pi / 8)]),
                                         np.array([0.410, -6.596 + 0.175 * math.sin(-5 * math.pi / 8),
                                                   -2.732 + 0.175 * math.cos(-5 * math.pi / 8)])],
                                '9.75': [np.array([0.410, -5.044 + 0.175 * math.sin(-3 * math.pi / 8),
                                                   2.089 + 0.175 * math.cos(-3 * math.pi / 8)]),
                                         np.array([0.410, -6.153 + 0.175 * math.sin(-3 * math.pi / 8),
                                                   2.549 + 0.175 * math.cos(-3 * math.pi / 8)]),
                                         np.array([0.410, -6.984 + 0.175 * math.sin(-3 * math.pi / 8),
                                                   2.893 + 0.175 * math.cos(-3 * math.pi / 8)])],
                                '11.25': [np.array([0.410, -1.871 + 0.175 * math.sin(-math.pi / 8),
                                                    4.518 + 0.175 * math.cos(-math.pi / 8)]),
                                          np.array([0.410, -2.417 + 0.175 * math.sin(-math.pi / 8),
                                                    5.834 + 0.175 * math.cos(-math.pi / 8)]),
                                          np.array([0.410, -2.761 + 0.175 * math.sin(-math.pi / 8),
                                                    6.666 + 0.175 * math.cos(-math.pi / 8)])]}
        self.cutter_rotmat_dict = {'0': [np.eye(3),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         np.eye(3),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         np.eye(3),
                                         np.eye(3),
                                         np.eye(3),
                                         np.eye(3),
                                         np.eye(3),
                                         np.eye(3)],
                                   '6': [rm.rotmat_from_euler(0, math.pi, 0),
                                         np.eye(3),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         np.eye(3),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         rm.rotmat_from_euler(0, math.pi, 0),
                                         rm.rotmat_from_euler(0, math.pi, 0)],
                                   '3': [rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0)],
                                   '9': [rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(-math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, 0, 0),
                                         rm.rotmat_from_euler(math.pi / 2, -math.pi / 6, 0, 'rxyz')],
                                   '1.5': [rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(3 * math.pi / 4, 0, 0)],
                                   '4.5': [rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(math.pi / 4, 0, 0)],
                                   '7.5': [rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, 0, 0),
                                           rm.rotmat_from_euler(-math.pi / 4, math.pi / 4, 0, 'rxyz')],
                                   '10.5': [rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 4, 0, 0)],
                                   '0.75': [rm.rotmat_from_euler(7 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(7 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(7 * math.pi / 8, 0, 0)],
                                   '2.25': [rm.rotmat_from_euler(5 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(5 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(5 * math.pi / 8, 0, 0)],
                                   '3.75': [rm.rotmat_from_euler(3 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(3 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(3 * math.pi / 8, 0, 0)],
                                   '5.25': [rm.rotmat_from_euler(math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(math.pi / 8, 0, 0)],
                                   '6.75': [rm.rotmat_from_euler(-math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-math.pi / 8, 0, 0)],
                                   '8.25': [rm.rotmat_from_euler(-3 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-3 * math.pi / 8, 0, 0)],
                                   '9.75': [rm.rotmat_from_euler(-5 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-5 * math.pi / 8, 0, 0),
                                            rm.rotmat_from_euler(-5 * math.pi / 8, 0, 0)],
                                   '11.25': [rm.rotmat_from_euler(-7 * math.pi / 8, 0, 0),
                                             rm.rotmat_from_euler(-7 * math.pi / 8, 0, 0),
                                             rm.rotmat_from_euler(-7 * math.pi / 8, 0, 0)]}
        self.cutters = {}
        for k in self.cutter_pos_dict.keys():
            self.cutters[k] = []
            for i, pos in enumerate(self.cutter_pos_dict[k]):
                tmp_jlc = jl.JLChain(pos=pos,
                                     rotmat=self.cutter_rotmat_dict[k][i],
                                     home_conf=np.zeros(1),
                                     name='cutter_' + str(k) + '_' + str(i))
                tmp_jlc.jnts[1]['loc_pos'] = np.zeros(3)
                tmp_jlc.jnts[1]['loc_motionax'] = np.array([0, 0, 1])
                tmp_jlc.lnks[1]['name'] = 'cutter'
                tmp_jlc.lnks[1]['loc_pos'] = np.array([0, 0, 0])
                tmp_jlc.lnks[1]['mesh_file'] = os.path.join(this_dir, 'meshes', 'cutter.stl')
                tmp_jlc.lnks[1]['rgba'] = [1, 1, .0, 1.0]
                tmp_jlc.tgtjnts = [1]
                tmp_jlc.finalize()
                self.cutters[k].append(tmp_jlc)

    def fix_to(self, pos, rotmat):
        self.pos = pos
        self.rotmat = rotmat
        self.head.fix_to(self.pos, self.rotmat)
        for k in self.cutters.keys():
            for i, cutter in enumerate(self.cutters[k]):
                new_pos = self.head.jnts[1]['gl_posq'] + self.head.jnts[1]['gl_rotmatq'].dot(
                    self.cutter_pos_dict[k][i])
                new_rotmat = self.head.jnts[1]['gl_posq'] + self.head.jnts[1]['gl_rotmatq'].dot(
                    self.cutter_rotmat_dict[k][i])
                cutter.fix_to(pos=new_pos, rotmat=new_rotmat)

    def fk(self, jnt_values=np.zeros(1)):
        """
        :param jnt_values: 7 or 3+7, 3=agv, 7=arm, 1=grpr; metrics: meter-radian
        :param component_name: 'arm', 'agv', or 'all'
        :return:
        author: weiwei
        date: 20201208toyonaka
        """
        self.head.fk(joint_values=jnt_values)
        for k in self.cutters.keys():
            for i, cutter in enumerate(self.cutters[k]):
                new_pos = self.head.jnts[1]['gl_posq'] + self.head.jnts[1]['gl_rotmatq'].dot(
                    self.cutter_pos_dict[k][i])
                new_rotmat = self.head.jnts[1]['gl_posq'] + self.head.jnts[1]['gl_rotmatq'].dot(
                    self.cutter_rotmat_dict[k][i])
                cutter.fix_to(pos=new_pos, rotmat=new_rotmat)

    def get_jnt_values(self):
        return self.head.get_jnt_values()

    def gen_stickmodel(self,
                       tcp_jntid=None,
                       tcp_loc_pos=None,
                       tcp_loc_rotmat=None,
                       toggle_tcp_frame=False,
                       toggle_jnt_frame=False,
                       toggle_connjnt=False,
                       name='tbm'):
        stickmodel = mc.ModelCollection(name=name)
        self.head.gen_stickmodel(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcp_frame=False,
                                 toggle_jnt_frame=toggle_jnt_frame).attach_to(stickmodel)
        for k in self.cutters.keys():
            for cutter in self.cutters[k]:
                cutter.gen_stickmodel(toggle_tcp_frame=False, toggle_jnt_frames=toggle_jnt_frame).attach_to(stickmodel)
        return stickmodel

    def gen_meshmodel(self,
                      tcp_jntid=None,
                      tcp_loc_pos=None,
                      tcp_loc_rotmat=None,
                      toggle_tcp_frame=False,
                      toggle_jnt_frame=False,
                      rgba=None,
                      name='tbm'):
        meshmodel = mc.ModelCollection(name=name)
        self.head.gen_mesh_model(tcp_loc_pos=None,
                                 tcp_loc_rotmat=None,
                                 toggle_tcp_frame=False,
                                 toggle_jnt_frame=toggle_jnt_frame,
                                 rgba=rgba).attach_to(meshmodel)
        for k in self.cutters.keys():
            for cutter in self.cutters[k]:
                cutter.gen_mesh_model(tcp_loc_pos=None,
                                      tcp_loc_rotmat=None,
                                      toggle_tcp_frame=False,
                                      toggle_jnt_frame=toggle_jnt_frame,
                                      rgba=rgba).attach_to(meshmodel)
        return meshmodel


if __name__ == '__main__':
    from wrs import basis as rm, robot_sim as jl, modeling as gm
    import wrs.visualization.panda.world as wd

    base = wd.World(cam_pos=[50, 0, 10], lookat_pos=[0, 0, 0])

    gm.gen_frame().attach_to(base)
    otherbot_s = TBM()
    # otherbot_s.fk(jnt_values=np.array([math.pi / 3]))
    otherbot_s.gen_meshmodel().attach_to(base)
    base.run()
