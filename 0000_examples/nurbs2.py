from geomdl import construct
from geomdl import fitting
from geomdl.visualization import VisMPL as vis
import numpy as np
import modeling.geometric_model as gm
import visualization.panda.world as wd
import basis.robot_math as rm
import math
from scipy.interpolate import RBFInterpolator
import vision.depth_camera.rbf_surface as rbfs

base = wd.World(cam_pos=np.array([.5,.1,.3]), lookat_pos=np.array([0,0,0.05]))
gm.gen_frame().attach_to(base)
tube_model = gm.GeometricModel(initor="./objects/bowl.stl")
tube_model.set_rgba([.3,.3,.3,.3])
tube_model.attach_to(base)
points, pfid = tube_model.sample_surface(radius=.002, nsample=10000)
points_normals = tube_model.objtrm.face_normals[pfid]
sampled_points = []
for id, p in enumerate(points.tolist()):
    if np.dot(np.array([1,0,0]), points_normals[id]) > .3 and p[0]>0:
        gm.gen_sphere(pos=p, radius=.001).attach_to(base)
        sampled_points.append(p)

# x - v
# y - u
rotmat_uv = rm.rotmat_from_euler(0, math.pi/2, 0)
sampled_points = rotmat_uv.dot(np.array(sampled_points).T).T
surface = rbfs.RBFSurface(sampled_points[:, :2], sampled_points[:,2])
surface_gm = surface.get_mesh()
surface_gm.set_rotmat(rotmat_uv.T)
surface_gm.attach_to(base)
base.run()