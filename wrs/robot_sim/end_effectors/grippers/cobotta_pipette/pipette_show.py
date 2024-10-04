
if __name__ == '__main__':
    import numpy as np
    import wrs.visualization.panda.world as wd
    from wrs import modeling as gm

    base = wd.World(cam_pos=[1, 1, 1], lookat_pos=[0, 0, .1])
    pm_s = gm.GeometricModel("./meshes/p1000g.stl")
    pm_s.attach_to(base)
    pm_b_s = gm.GeometricModel("./meshes/p1000g_body.stl")
    pm_b_s.set_scale(scale=[1.03,1.03,1.01])
    pm_b_s.set_pos(np.array([0,0, 0.1463]))
    pm_b_s.set_rgba(rgba=[.3, .4, .6, 1])
    pm_b_s.attach_to(base)
    base.run()
