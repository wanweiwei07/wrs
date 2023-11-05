import numpy as np
import copy
import scipy.signal as ss

class Node(object):

    def __init__(self, state):
        """
        :param state: np.array nrow*ncolumn
        author: weiwei
        date: 20190828, 20200104
        """
        self.state = copy.deepcopy(state)
        self._nrow, self._ncolumn = self.state.shape
        self.state_size = self._nrow*self._ncolumn
        self.parent = None
        self.past_cost = 0
        self.state_padded = np.pad(self.state, (1,1), 'constant')

    def __getitem__(self, x):
        return self.state[x]

    def __eq__(self, another_node):
        """
        determine if two nodes are the same
        :return:
        author: weiwei
        date: 20190828
        """
        return np.array_equal(self.state, another_node.state)

    def __repr__(self):
        """
        overload the printed results
        :return:
        author: weiwei
        date: 20191003
        """
        outstring = "["
        for i in range(self._nrow):
            if i == 0:
                outstring += "["
            else:
                outstring += " ["
            for j in range(self._ncolumn):
                outstring = outstring+str(self.state[i][j])+","
            outstring = outstring[:-1] + "]"
            outstring += ",\n"
        outstring = outstring[:-2] + "]]"
        return outstring

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncolumn(self):
        return self._ncolumn

    def get_3x3(self, i, j):
        """
        get the surrounding 3x3 mat at i,j
        :param i:
        :param j:
        :return:
        author: weiwei
        date: 20200425
        """
        i = i+1
        j = j+1
        return self.self.state_padded[i-1:i+1, j-1:j+1]

class TubePuzzle(object):

    def __init__(self, init_state, goal_pattern = None):
        """
        :param nrow:
        :param ncolumn:
        :param init_state: nrow*ncolumn int array, tube id starts from 1, maximum 4
        author: weiwei
        date: 20191003
        """
        self._nrow = init_state.shape[0]
        self._ncolumn = init_state.shape[1]
        self.init_state = np.zeros((self._nrow, self._ncolumn), dtype="int")
        self.open_list = []
        self.close_list = []
        self._set_init_state(init_state)
        if goal_pattern is None:
            self.goal_pattern = np.array([[1,1,1,1,0,0,2,2,2,2],
                                          [1,1,1,1,0,0,2,2,2,2],
                                          [1,1,1,1,0,0,2,2,2,2],
                                          [1,1,1,0,0,0,0,2,2,2],
                                          [1,1,1,0,0,0,0,2,2,2]])
        else:
            self.goal_pattern = goal_pattern

    def _set_init_state(self, init_state):
        """
        change the elements of the puzzle using state
        :param init_state: 2d array
        :return:
        author: weiwei
        date: 20190828, 20200104osaka, 20210914osaka
        """
        if init_state.shape != (self._nrow, self._ncolumn):
            print("Wrong number of elements in elelist!")
            raise Exception("Number of elements error!")
        self.init_state = init_state

    def _heuristics(self, node):
        """
        heuristics
        :return:
        author: weiwei
        date: 20200104
        """
        return np.sum((self.goal_pattern!=1)*(node.state==1)+(self.goal_pattern!=2)*(node.state==2))

    def isdone(self, node):
        """
        :return:
        author: weiwei
        date: 20190828
        """
        if np.any((self.goal_pattern != 1)*(node.state==1)) or np.any((self.goal_pattern != 2)*(node.state==2)):
            return False
        return True

    def f_cost(self, node):
        hs = self._heuristics(node)
        gs = node.past_cost
        return hs+gs, hs, gs

    def get_movable_fillable_pair(self, node):
        """
        get a list of movable and fillable pairs
        :param node see Node
        :return: [[(i,j), (k,l)], ...]
        author: weiwei
        date: 20191003osaka, 20200104osaka
        """
        # filtering
        # mask_ulbr = np.array([[1,0,0],[0,0,0],[0,0,1]])
        # mask_urbl = np.array([[0,0,1],[0,0,0],[1,0,0]])
        # mask_ulbr2 = np.array([[1,0,0],[1,0,0],[0,1,1]])
        # mask_urbl2 = np.array([[0,1,1],[1,0,0],[1,0,0]])
        # mask_ulbr2_flp = np.array([[1,1,0],[0,0,1],[0,0,1]])
        # mask_urbl2_flp = np.array([[0,0,1],[0,0,1],[1,1,0]])
        mask_ucbc = np.array([[0,1,0],
                              [0,0,0],
                              [0,1,0]])
        mask_crcl = np.array([[0,0,0],
                              [1,0,1],
                              [0,0,0]])
        mask_ul = np.array([[1,1,1],
                            [1,0,0],
                            [1,0,0]])
        mask_ur = np.array([[1,1,1],
                            [0,0,1],
                            [0,0,1]])
        mask_bl = np.array([[1,0,0],
                            [1,0,0],
                            [1,1,1]])
        mask_br = np.array([[0,0,1],
                            [0,0,1],
                            [1,1,1]])
        # cg_ulbr = ss.correlate2d(node.state, mask_ulbr)[1:-1,1:-1]
        # cg_urbl = ss.correlate2d(node.state, mask_urbl)[1:-1,1:-1]
        # cg_ulbr2_flp = ss.correlate2d(node.state, mask_ulbr2_flp)[1:-1,1:-1]
        # cg_urbl2_flp = ss.correlate2d(node.state, mask_urbl2_flp)[1:-1,1:-1]
        # cg_ulbr2_flp = ss.correlate2d(node.state, mask_ulbr2_flp)[1:-1,1:-1]
        # cg_urbl2_flp = ss.correlate2d(node.state, mask_urbl2_flp)[1:-1,1:-1]
        cg_ucbc = ss.correlate2d(node.state, mask_ucbc)[1:-1,1:-1]
        cg_crcl = ss.correlate2d(node.state, mask_crcl)[1:-1,1:-1]
        cg_ul = ss.correlate2d(node.state, mask_ul)[1:-1,1:-1]
        cg_ur = ss.correlate2d(node.state, mask_ur)[1:-1,1:-1]
        cg_bl = ss.correlate2d(node.state, mask_bl)[1:-1,1:-1]
        cg_br = ss.correlate2d(node.state, mask_br)[1:-1,1:-1]
        ## fillable
        # cf = ((cg_ulbr==0)+(cg_urbl==0)+(cg_ulbr_flp==0)+(cg_urbl_flp==0)+(cg_ucbc==0)+(cg_crcl==0))*(state.grid==0)
        # cf = ((cg_ulbr==0)+(cg_urbl==0)+(cg_ucbc==0)+(cg_crcl==0))*(state.grid==0)
        # cf = ((cg_ucbc==0)+(cg_crcl==0))*(state.grid==0)
        cf = ((cg_ucbc==0)+(cg_crcl==0)+(cg_ul==0)+(cg_ur==0)+(cg_bl==0)+(cg_br==0))*(node.state==0)
        ## always fill the first element
        # # fillable 1
        # for i in range(np.asarray(np.where((self.goal_pattern==1)*cf)).T.shape[0]):
        #     fillable_type1 = [np.asarray(np.where((self.goal_pattern==1)*cf)).T[i]]
        #     if weight_array[fillable_type1[0][0], fillable_type1[0][1]] !=0:
        #         continue
        # # fillable 2
        # for i in range(np.asarray(np.where((self.goal_pattern==2)*cf)).T.shape[0]):
        #     fillable_type2 = [np.asarray(np.where((self.goal_pattern==2)*cf)).T[i]]
        #     if weight_array[fillable_type2[0][0], fillable_type2[0][1]] !=0:
        #         continue
        fillable_type1 = [np.asarray(np.where((self.goal_pattern==1)*cf)).T[0]]
        fillable_type2 = [np.asarray(np.where((self.goal_pattern==2)*cf)).T[0]]
        # # fillable 1
        # fillable_type1 = np.asarray(np.where((self.goal_pattern==1)*cf)).T
        # # fillable 2
        # fillable_type2 = np.asarray(np.where((self.goal_pattern==2)*cf)).T
        ## graspable
        # cg_ulbr[state.grid==0]=-1
        # cg_urbl[state.grid==0]=-1
        # cg_ulbr_flp[state.grid==0]=-1
        # cg_urbl_flp[state.grid==0]=-1
        cg_ucbc[node.state==0]=-1
        cg_crcl[node.state==0]=-1
        cg_ul[node.state==0]=-1
        cg_ur[node.state==0]=-1
        cg_bl[node.state==0]=-1
        cg_br[node.state==0]=-1
        # cg = (cg_ulbr==0)+(cg_urbl==0)+(cg_ulbr_flp==0)+(cg_urbl_flp==0)+(cg_ucbc==0)+(cg_crcl==0)
        # cg = (cg_ulbr==0)+(cg_urbl==0)+(cg_ucbc==0)+(cg_crcl==0)
        # cg = (cg_ucbc==0)+(cg_crcl==0)
        cg = (cg_ucbc==0)+(cg_crcl==0)+(cg_ul==0)+(cg_ur==0)+(cg_bl==0)+(cg_br==0)
        # movable 1
        movable_type1 = np.asarray(np.where(cg*(node.state==1))).T
        # movable 2
        movable_type2 = np.asarray(np.where(cg*(node.state==2))).T
        movable_expanded_type1 = np.repeat(movable_type1, len(fillable_type1), axis=0)
        movable_expanded_type2 = np.repeat(movable_type2, len(fillable_type2), axis=0)
        if len(movable_expanded_type1)==0:
            movable_elements = movable_expanded_type2
        elif len(movable_expanded_type2)==0:
            movable_elements = movable_expanded_type1
        else:
            movable_elements = np.concatenate((movable_expanded_type1, movable_expanded_type2), axis=0)
        fillable_expanded_type1 = np.tile(fillable_type1, (len(movable_type1),1))
        fillable_expanded_type2 = np.tile(fillable_type2, (len(movable_type2),1))
        if len(fillable_expanded_type1)==0:
            fillable_elements = fillable_expanded_type2
        elif len(fillable_expanded_type2)==0:
            fillable_elements = fillable_expanded_type1
        else:
            fillable_elements = np.concatenate((fillable_expanded_type1, fillable_expanded_type2), axis=0)
        return movable_elements, fillable_elements

    def _reorder_open_list(self):
        self.open_list.sort(key=lambda x: (self.f_cost(x)[0], self.f_cost(x)[1]))

    def astar_search(self, weight_array=None):
        """
        build a graph considering the movable and fillable ids
        :param weight_array
        :return:
        author: weiwei
        date: 20191003
        """
        if weight_array is None:
            weight_array = np.zeros_like(self.init_state)
        start_node = Node(self.init_state)
        self.open_list = [start_node]
        while True:
            # if len(self.openlist)>=2:
            #     for eachnode in self.openlist:
            #         print(eachnode)
            #         print(eachnode.fcost())
            #     print("\n")
            self._reorder_open_list()
            # for opennode in self.openlist:
            #     print(opennode)
            #     print(self.fcost(opennode))
            #     print("\n")
            self.close_list.append(self.open_list.pop(0))
            # movableids = self.getMovableIds(self.closelist[-1])
            # fillableids = self.getFillableIds(self.closelist[-1])
            # if len(movableids) == 0 or len(fillableids) == 0:
            #     print("No path found!")
            #     return []
            # for mid in movableids:
            #     for fid in fillableids:
            # todo consider weight array when get movable fillable pair
            movable_elements, fillable_elements = self.get_movable_fillable_pair(self.close_list[-1])
            print(movable_elements)
            print(fillable_elements)
            if movable_elements.shape[0] == 0:
                print("No path found!")
                # return []
            for i in range(movable_elements.shape[0]):
                mi, mj = movable_elements[i]
                fi, fj = fillable_elements[i]
                if weight_array[mi, mj] != 0 and weight_array[fi, fj] != 0:
                    continue
                tmp_node = copy.deepcopy(self.close_list[-1])
                tmp_node.parent = self.close_list[-1]
                tmp_node.past_cost = self.close_list[-1].past_cost+1
                tmp_node[fi][fj] = tmp_node[mi][mj]
                tmp_node[mi][mj] = 0
                #  check if path is found
                if self.isdone(tmp_node):
                    path = [tmp_node]
                    parent = tmp_node.parent
                    while parent is not None:
                        path.append(parent)
                        parent = parent.parent
                    print("Path found!")
                    # print(tmpelearray)
                    # print(self.fcost(tmpelearray))
                    # for eachnode in path:
                    #     print(eachnode)
                    return path[::-1]
                # check if in openlist
                flag_in_openlist = False
                for each_node in self.open_list:
                    if each_node == tmp_node:
                        flag_in_openlist = True
                        if self.f_cost(each_node)[0] <= self.f_cost(tmp_node)[0]:
                            pass
                            # no need to update position
                        else:
                            each_node.parent = tmp_node.parent
                            each_node.gs = tmp_node.gs
                            # self._reorderopenlist()
                        # continue
                        break
                if flag_in_openlist:
                    continue
                else:
                    # not in openlist append and sort openlist
                    self.open_list.append(tmp_node)

if __name__=="__main__":
    # down x, right y
    elearray = np.array([[1,0,0,0,1,0,0,0,0,0],
                         [0,0,0,0,0,0,0,2,0,2],
                         [0,0,0,0,0,0,0,0,2,0],
                         [1,0,0,0,0,0,0,0,2,2],
                         [1,0,0,0,0,0,0,2,0,2]])
    elearray = np.array([[0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,0,0,0,0,0,0,0],
                         [2,2,0,2,1,0,0,0,0,0],
                         [1,1,0,1,2,0,0,0,0,2],
                         [0,2,0,0,0,0,0,0,0,2]])
    elearray = np.array([[0,0,0,0,0,0,0,0,0,0],
                         [0,0,0,2,2,2,2,0,0,0],
                         [0,0,2,1,1,1,0,0,0,0],
                         [0,0,2,1,2,2,0,0,0,0],
                         [0,0,0,0,2,0,0,0,0,0]])
    tp = TubePuzzle(elearray)
    # trm_primit.getMovableIds(Node(state))
    # print(Node(state).fcost())
    # print(trm_primit.fcost(Node(state)))
    path = tp.astar_search()
    # for state in path:////////////
    #     print(state)
