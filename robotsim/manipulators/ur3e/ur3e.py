if __name__ == '__main__':
    import time
    import visualization.panda.world as wd
    import modeling.geometricmodel as gm

    base = wd.World(camp=[2, 0, 1], lookatpos=[0, 0, 0.5])
    basemodel = gm.GeometricModel("./stl/base.dae")
    basemodel.attach_to(base)
    base.run()
