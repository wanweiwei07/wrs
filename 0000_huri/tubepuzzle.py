import numpy as np
import copy
import math

class Node(object):

    def __init__(self, grid):

        """

        :param grid: np.array nrow*ncolumn

        author: weiwei
        date: 20190828
        """

        grid = copy.deepcopy(grid)
        self.grid = grid
        self._nrow = grid.shape[0]
        self._ncolumn = grid.shape[1]
        self.ngrids = self._nrow*self._ncolumn
        self.tubes = np.unique(self.grid)[1:]
        self.ntubes = self.tubes.shape[0]
        if self.ntubes > 5:
            print("We do not allow more than 4 types of tubes!")
        elif self.ntubes == 4:
            self.boundids = [[0,1,10,11,20,21,30,31,40,41], [3,4,13,14,23,24,33,34,43,44],
                             [6,7,16,17,26,27,36,37,46,47], [8,9,18,19,28,29,38,39,48,49]]
        elif self.ntubes == 3:
            self.boundids = [[0,1,2,10,11,12,20,21,22,30,31,32,40,41,42],
                              [4,5,6,14,15,16,24,25,26,34,35,36,44,45,46],
                              [7,8,9,17,18,19,27,28,29,37,38,39,47,48,49]]
        elif self.ntubes == 2:
            self.boundids = [[0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33,40,41,42,43],
                              [6,7,8,9,16,17,18,19,26,27,28,29,36,37,38,39,46,47,48,49]]
        elif self.ntubes < 2:
            print("We require at least 2 types of tubes!")
        self.parent = None

    @property
    def nrow(self):
        return self._nrow

    @property
    def ncolumn(self):
        return self._ncolumn

    def __getitem__(self, x):
        return self.grid[x]

    def __eq__(self, anothernode):
        """
        determine if two nodes are the same
        :return:

        author: weiwei
        date: 20190828
        """

        return np.array_equal(self.grid, anothernode.grid)

    def _aitoli(self, i, j):
        """
        convert arrayindex to list index

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        return i*self._ncolumn+j

    def _litoai(self, i):
        """
        convert arrayindex to list index

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        return [np.floor(i/self._ncolumn), i%self._ncolumn]

    def isdone(self):
        """

        :return:

        author: weiwei
        date: 20190828
        """

        gridflatten = self.grid.flatten()
        for i, tubeid in enumerate(self.tubes):
            tubegridids = np.where(gridflatten==tubeid)[0]
            ndiff = len(np.setdiff1d(tubegridids, self.boundids[i]))
            if ndiff > 0:
                return False
            # for id in tubegridids:
            #     if id not in self.boundids[i]:
            #         return False
            # isdone = isdone and (np.all(tubegridids<=self.ngrids*(i+1)/self.ntubes) and np.all(tubegridids>=self.ngrids*i/self.ntubes))
        return True

    def _hs(self):
        """
        heuristics

        :return:

        author: weiwei
        date: 20190828
        """

        ndiff = 0
        gridflatten = self.grid.flatten()
        for i, tubeid in enumerate(self.tubes):
            tubegridids = np.where(gridflatten==tubeid)[0]
            # ndiff = ndiff+np.where(tubegridids>self.ngrids*(i+1)/self.ntubes)[0].shape[0]+np.where(tubegridids<self.ngrids*(i)/self.ntubes)[0].shape[0]
            # ndiff = ndiff+np.where(tubegridids<i*(math.ceil(self._nrow/self.ntubes)+1)*self._ncolumn)[0].shape[0]+np.where(tubegridids>i*(math.ceil(self._nrow/self.ntubes)+1)*self._ncolumn)[0].shape[0]
            # print(tubegridids)
            # print(self.boundids[i])
            ndiff += len(np.setdiff1d(tubegridids, self.boundids[i]))
            # for id in tubegridids:
            #     if id not in self.boundids[i]:
            #         ndiff += 1
        return ndiff

    def fcost(self):
        hs = self._hs()
        gs = 0
        parent = self.parent
        while parent is not None:
            gs += 1
            parent = parent.parent
        return hs+gs, hs, gs

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
                outstring = outstring+str(self.grid[i][j])+","
            outstring = outstring[:-1] + "]"
            outstring += ",\n"
        outstring = outstring[:-2] + "]]"

        return outstring

class TubePuzzle(object):

    def __init__(self, elearray):
        """

        :param nrow:
        :param ncolumn:
        :param elearray: nrow*ncolumn int array, tube id starts from 1, maximum 4

        author: weiwei
        date: 20191003
        """

        self._nrow = elearray.shape[0]
        self._ncolumn = elearray.shape[1]
        self.elearray = np.zeros((self._nrow, self._ncolumn), dtype="int")
        self.openlist = []
        self.closelist = []
        self._setValues(elearray)
        self.tubes = np.unique(self.elearray)[1:]
        self.ntubes = self.tubes.shape[0]
        if self.ntubes > 5:
            print("We do not allow more than 4 types of tubes!")
        elif self.ntubes == 4:
            self.boundids = [[0,1,10,11,20,21,30,31,40,41], [3,4,13,14,23,24,33,34,43,44],
                             [6,7,16,17,26,27,36,37,46,47], [8,9,18,19,28,29,38,39,48,49]]
        elif self.ntubes == 3:
            self.boundids = [[0,1,2,10,11,12,20,21,22,30,31,32,40,41,42],
                              [4,5,6,14,15,16,24,25,26,34,35,36,44,45,46],
                              [7,8,9,17,18,19,27,28,29,37,38,39,47,48,49]]
        elif self.ntubes == 2:
            self.boundids = [[0,1,2,3,10,11,12,13,20,21,22,23,30,31,32,33,40,41,42,43],
                              [6,7,8,9,16,17,18,19,26,27,28,29,36,37,38,39,46,47,48,49]]
        elif self.ntubes == 2:
            print("We require at least 2 types of tubes!")

    def _aitoli(self, i, j):
        """
        convert arrayindex to list index

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        return i*self._ncolumn+j

    def _litoai(self, i):
        """
        convert arrayindex to list index

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        return [np.int(np.floor(i/self._ncolumn)), i%self._ncolumn]

    def _eleexist(self, i, j):
        """
        determine if ele (i,j) exists or not

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        if (i>=0 and i<self._nrow) and (j>=0 and j<self._ncolumn):
            return True
        else:
            return False

    def _occupied(self, node, i, j):
        """
        determine if ele(i,j) is occupied or not

        :param i:
        :param j:
        :return:

        author: weiwei
        date: 20190828
        """

        if node[i][j] == 0:
            return False
        else:
            return True

    def _isclear(self, node, i, j):
        """
        ele(i,j) is empty if it is outside the array or it is zero


        :param i:
        :param j:
        :return:
        """

        if (not self._eleexist(i,j)) or (not self._occupied(node, i,j)):
            return 1
        else:
            return 0


    def _setValues(self, elearray):
        """
        change the elements of the puzzle using state

        :param elearray: 2d array
        :return:

        author: weiwei
        date: 20190828
        """

        if elearray.shape != (self._nrow, self._ncolumn):
            print("Wrong number of elements in elelist!")
            raise Exception("Number of elements error!")

        for i in range(self._nrow):
            for j in range(self._ncolumn):
                self.elearray[i][j] = elearray[i][j]

    def getMovableIds(self, node):
        """
        get a list of tubes that are movable

        :param node see Node
        :return:

        author: weiwei
        date: 20190828
        """

        returnlist = []
        for i in range(self._nrow):
            for j in range(self._ncolumn):
                if node[i][j] > 0:
                    iup = i-1
                    jup = j
                    idown = i+1
                    jdown = j
                    ileft = i
                    jleft = j-1
                    iright = i
                    jright = j+1
                    nclear = self._isclear(node, iup, jup)+\
                             self._isclear(node, idown, jdown)+\
                             self._isclear(node, ileft, jleft)+\
                             self._isclear(node, iright, jright)
                    if nclear >= 2:
                        returnlist.append((i,j))
        return returnlist

    def getFillableIds(self, node):
        """
        get a list of grids that are fillable

        :param node see Node
        :return:

        author: weiwei
        date: 20190828
        """

        returnlist = []
        for i in range(self._nrow):
            for j in range(self._ncolumn):
                if node[i][j] == 0 and (i in [0,4]):
                    iup = i-1
                    jup = j
                    idown = i+1
                    jdown = j
                    ileft = i
                    jleft = j-1
                    iright = i
                    jright = j+1
                    nclear = self._isclear(node, iup, jup)+\
                             self._isclear(node, idown, jdown)+\
                             self._isclear(node, ileft, jleft)+\
                             self._isclear(node, iright, jright)
                    if nclear >= 2:
                        returnlist.append((i,j))
        return returnlist

    def getMovableFillablePair(self, node):
        """
        get a list of movable and fillable pairs

        :param node see Node
        :return: [[(i,j), (k,l)], ...]

        author: weiwei
        date: 20191003
        """

        fillablegridsids = []
        for bid in range(len(self.boundids)):
            fillableinbounds = []
            for id in self.boundids[bid]:
                i, j = self._litoai(id)
                if node[i][j] == 0:
                    iup = i-1
                    jup = j
                    idown = i+1
                    jdown = j
                    ileft = i
                    jleft = j-1
                    iright = i
                    jright = j+1
                    nclear1 = self._isclear(node, iup, jup)+\
                             self._isclear(node, idown, jdown)
                    nclear2 = self._isclear(node, ileft, jleft)+\
                             self._isclear(node, iright, jright)
                    nclear3 = self._isclear(node, iup, jright)+\
                             self._isclear(node, idown, jleft)
                    nclear4 = self._isclear(node, iup, jleft)+\
                             self._isclear(node, idown, jright)
                    # nclear = self._isclear(node, iup, jup)+\
                    #          self._isclear(node, idown, jdown)+\
                    #          self._isclear(node, ileft, jleft)+\
                    #          self._isclear(node, iright, jright)
                    if nclear1 == 2 or nclear2 == 2 or nclear3 == 2 or nclear4 == 2:
                        fillableinbounds.append((i,j))
            fillablegridsids.append(fillableinbounds)

        returnlist = []
        for i in range(self._nrow):
            for j in range(self._ncolumn):
                if node[i][j] > 0:
                    iup = i-1
                    jup = j
                    idown = i+1
                    jdown = j
                    ileft = i
                    jleft = j-1
                    iright = i
                    jright = j+1
                    nclear1 = self._isclear(node, iup, jup)+\
                             self._isclear(node, idown, jdown)
                    nclear2 = self._isclear(node, ileft, jleft)+\
                             self._isclear(node, iright, jright)
                    nclear3 = self._isclear(node, iup, jright)+\
                             self._isclear(node, idown, jleft)
                    nclear4 = self._isclear(node, iup, jleft)+\
                             self._isclear(node, idown, jright)
                    # nclear = self._isclear(node, iup, jup)+\
                    #          self._isclear(node, idown, jdown)+\
                    #          self._isclear(node, ileft, jleft)+\
                    #          self._isclear(node, iright, jright)
                    if nclear1 == 2 or nclear2 == 2 or nclear3 == 2 or nclear4 == 2:
                        for grid in fillablegridsids[node[i][j]-1]:
                            returnlist.append(((i,j), grid))
        return returnlist

    def _reorderopenlist(self):
        self.openlist.sort(key=lambda x: (x.fcost()[0], x.fcost()[1]))

    def atarSearch(self):
        """

        build a graph considering the movable and fillable ids

        :return:

        author: weiwei
        date: 20191003
        """

        startnode = Node(self.elearray)
        self.openlist = [startnode]
        while True:
            # if len(self.openlist)>=2:
            #     for eachnode in self.openlist:
            #         print(eachnode)
            #         print(eachnode.fcost())
            #     print("\n")
            print(self.openlist[0])
            print(self.openlist[0].fcost())
            print("\n")
            self.closelist.append(self.openlist.pop(0))
            # movableids = self.getMovableIds(self.closelist[-1])
            # fillableids = self.getFillableIds(self.closelist[-1])
            # if len(movableids) == 0 or len(fillableids) == 0:
            #     print("No path found!")
            #     return []
            # for mid in movableids:
            #     for fid in fillableids:

            mfpairs = self.getMovableFillablePair(self.closelist[-1])
            if len(mfpairs) == 0:
                print("No path found!")
                return []
            for mfp in mfpairs:
                mi, mj = mfp[0]
                fi, fj = mfp[1]
                tmpelearray = copy.deepcopy(self.closelist[-1])
                tmpelearray.parent = self.closelist[-1]
                tmpelearray[fi][fj] = tmpelearray[mi][mj]
                tmpelearray[mi][mj] = 0
                #  check if path is found
                if tmpelearray.isdone():
                    path = [tmpelearray]
                    parent = tmpelearray.parent
                    while parent is not None:
                        path.append(parent)
                        parent = parent.parent
                    print("Path found!")
                    # for eachnode in path:
                    #     print(eachnode)
                    return path[::-1]
                # check if in openlist
                for eachnode in self.openlist:
                    if eachnode == tmpelearray:
                        if eachnode.fcost()[0] <= tmpelearray.fcost()[0]:
                            pass
                            # no need to update position
                        else:
                            eachnode.parent = tmpelearray.parent
                            self._reorderopenlist()
                        continue
                # not in openlist append and sort openlist
                self.openlist.append(tmpelearray)
                self._reorderopenlist()


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
    path = tp.atarSearch()
    for node in path:
        print(node)
