import numpy as np
import networkx as nx

class RoadMap(object):

    def __init__(self):
        self.nxg = nx.Graph()
        # list for easy access
        self.node_nid_list = []
        self.node_configuration_list = []

    def get_node(self, nid):
        return self.nxg.nodes[nid]

    def get_node_configuration(self, nid):
        return self.nxg.nodes[nid]['conf']

    def get_node_cost(self, nid):
        return self.nxg.nodes[nid]['cost']

    def add_node(self, nid, configuration):
        """
        :param nid:
        :param configuration: 1xn nparray
        :return:
        author: weiwei
        date: 20210522
        """
        self.nxg.add_node(nid, conf=configuration)
        self.node_confs.append(configuration)

    def add_edge(self, nid0, nid1):
        self.nxg.add_edge(nid0, nid1)

    def find_shortest_path(self, nid0, nid1):
        return nx.shortest_path(self.nxg, nid0, nid1)

    def find_nearest_nid(self, conf0):
        diff_array = conf0-np.array(self.node_configuration_list)
        nid = np.argmin(np.linalg.norm(diff_array, 1))