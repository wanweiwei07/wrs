import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import math
import pickle
import modeling.collision_model as cm
# import vision.depth_camera.surface.gaussian_surface as gs
# import vision.depth_camera.surface.quadrantic_surface as qs
import vision.depth_camera.surface.bibspline_surface as bs
import vision.depth_camera.surface.plane_surface as ps
import vision.depth_camera.surface.rbf_surface as rbfs

base = wd.World(cam_pos=np.array([.5,.1,.3]), lookat_pos=np.array([0,0,0.02]))
model_pcd = pickle.load(open("helmet_gaussian.pkl", "rb"))['objpcd'] * 1e-3
origin = np.mean(model_pcd, axis=0)
bowl_samples = model_pcd - origin
bowl_model = cm.CollisionModel(initor=bowl_samples)
bowl_model.attach_to(base)
# base.run()
# sampled_points = []
# for id, p in enumerate(points.tolist()):
#     if np.dot(np.array([1,0,0]), points_normals[id]) > .3 and p[0]>0:
#         gm.gen_sphere(pos=p, radius=.001).attach_to(base)
#         sampled_points.append(p)

# x - v
# y - u
# rotmat_uv = rm.rotmat_from_euler(0, math.pi/2, 0)
# sampled_points = rotmat_uv.dot(np.array(sampled_points).T).T
# surface = rbfs.RBFSurface(bowl_samples[:, :2], bowl_samples[:,2])
# surface = gs.MixedGaussianSurface(sampled_points[:, :2], sampled_points[:,2], n_mix=1)
# surface = qs.QuadraticSurface(sampled_points[:, :2], sampled_points[:,2])
surface = bs.BiBSpline(bowl_samples[:, :2], bowl_samples[:,2])
# surface = ps.PlaneSurface(bowl_samples[:,:2], bowl_samples[:,2])
surface_gm = surface.get_gometricmodel([[-.3,.3],[-.3,.3]])
surface_gm.attach_to(base)
base.run()