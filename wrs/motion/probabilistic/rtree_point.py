import numpy as np
from rtree import index

class RtreePoint():

    def __init__(self, dimension):
        p = index.Property()
        p.dimension = dimension
        p.storage= index.RT_Memory
        self._idx_rtp = index.Index(properties = p)
        self._dimension = dimension

    def insert(self, id, point):
        """
        The dimension of a point must be equal to dimension of the tree
        :param id
        :param point: a 1xn array
        :return:
        author: weiwei
        date: 20180520
        """
        if id == 'start':
            id = -1
        if id == 'goal':
            id = -2
        self._idx_rtp.insert(id, np.hstack((point, point)), point)

    def nearest(self, point):
        """
        The dimension of a point must be equal to dimension of the tree
        :param point: a 1xn list
        :return: id of the neareast point (use 'raw' for array; use True for object
        author: weiwei
        date: 20180520
        """
        return_id = list(self._idx_rtp.nearest(np.hstack((point, point)), 1, objects=False))[0]
        if return_id == -1:
            return_id = 'start'
        if return_id == -2:
            return_id = 'goal'
        return return_id