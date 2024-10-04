import numpy as np
import wrs.visualization.panda.world as wd
from wrs import basis as rm, vision as ps, modeling as gm
import math
# import vision.depth_camera.surface.gaussian_surface as gs
# import vision.depth_camera.surface.quadrantic_surface as qs

base = wd.World(cam_pos=np.array([.5,.1,.3]), lookat_pos=np.array([0,0,0.05]))
gm.gen_frame().attach_to(base)
tube_model = gm.GeometricModel(initor="./objects/bowl.stl")
tube_model.set_rgba([.3,.3,.3,.3])
tube_model.attach_to(base)
points, points_normals = tube_model.sample_surface(radius=.002, n_samples=10000, toggle_option='normals')
sampled_points = []
for id, p in enumerate(points.tolist()):
    if np.dot(np.array([1,0,0]), points_normals[id]) > .3 and p[0]>0:
        gm.gen_sphere(pos=p, radius=.001).attach_to(base)
        sampled_points.append(p)

# x - v
# y - u
rotmat_uv = rm.rotmat_from_euler(0, math.pi/2, 0)
sampled_points = rotmat_uv.dot(np.array(sampled_points).T).T
# surface = rbfs.RBFSurface(sampled_points[:, :2], sampled_points[:,2])
# surface = gs.MixedGaussianSurface(sampled_points[:, :2], sampled_points[:,2], n_mix=1)
# surface = qs.QuadraticSurface(sampled_points[:, :2], sampled_points[:,2])
# surface = bs.BiBSpline(sampled_points[:, :2], sampled_points[:,2])
surface = ps.PlaneSurface(sampled_points[:,:2], sampled_points[:,2])
surface_gm = surface.get_gometricmodel()
surface_gm.set_rotmat(rotmat_uv.T)
surface_gm.attach_to(base)
base.run()