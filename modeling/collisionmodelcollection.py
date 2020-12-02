from panda3d.core import NodePath

class CollisionModelCollection(object):
    """
    a helper class to further hide pandanodes
    list of collision models can be reparented to this collection for visualization
    author: weiwei
    date: 201900825
    """

    def __init__(self, name = "cmcollection"):
        self.__name = name
        self.__pdnp = NodePath(name)
        self.__cmlist = []

    @property
    def name(self):
        # read-only property
        return self.__name

    @property
    def objnp(self):
        # read-only property
        return self.__pdnp

    @property
    def cmlist(self):
        # read-only property
        return self.__cmlist

    def addcm(self, objcm):
        objcm.reparentTo(self.__pdnp)
        self.__cmlist.append(objcm)

    def showcn(self):
        for cm in self.__cmlist:
            cm.showcn()

    def unshowcn(self):
        for cm in self.__cmlist:
            cm.unshowcn()

    def reparentTo(self, pandanp):
        self.__pdnp.reparentTo(pandanp)


