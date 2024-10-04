import numpy as np
from wrs import basis as rm
import pickle
from scipy.spatial import cKDTree
# import vision.depth_camera.surface.gaussian_surface as gs
# import vision.depth_camera.surface.rbf_surface as rbfs
import pandaplotutils.pandactrl as pc
import pcd_utils as pcdu

base = pc.World(camp=np.array([-300, -900, 300]), lookatpos=np.array([0, 0, 0]))
# mgm.gen_frame().attach_to(base)
model_pcd = pickle.load(open("..\log\mapping\helmet_gaussian.pkl", "rb"))['objpcd']
origin = np.mean(model_pcd, axis=0)
bowl_samples = model_pcd - origin
pcdu.show_pcd(bowl_samples,rgba=(.3, .3, .3, .3))
# bowl_model = mcm.CollisionModel(initializer=bowl_samples)
# bowl_model.set_rgba([.3, .3, .3, .3])
# bowl_model.set_rotmat(rm.rotmat_from_euler(math.pi,0,0))
# bowl_model.attach_to(base)
# print(model_pcd)
# bowl_model.attach_to(base)
# base.run()

tree = cKDTree(bowl_samples)
point_id = 7000
nearby_sample_ids = tree.query_ball_point(bowl_samples[point_id, :], 10)
nearby_samples = bowl_samples[nearby_sample_ids]
colors = np.tile(np.array([1, 0, 0,1]), (len(nearby_samples),1))
print(nearby_samples.shape)
print(colors.shape)
# nearby_samples_withcolor = np.column_stack((nearby_samples, colors))
# mgm.GeometricModel(nearby_samples_withcolor).attach_to(base)
pcdu.show_pcd(nearby_samples,rgba=(1, 0, 0,1))

plane_center, plane_normal = rm.fit_plane(nearby_samples)
plane_tangential = rm.orthogonal_vector(plane_normal)
plane_tmp = np.cross(plane_normal, plane_tangential)
plane_rotmat = np.column_stack((plane_tangential, plane_tmp, plane_normal))
nearby_samples_on_xy = plane_rotmat.T.dot((nearby_samples - plane_center).T).T
surface = gs.MixedGaussianSurface(nearby_samples_on_xy[:, :2], nearby_samples_on_xy[:, 2], n_mix=1)
# t_npt_on_xy = plane_rotmat.T.dot(t_npt - plane_center)
# projected_t_npt_z_on_xy = surface.get_zdata(np.array([t_npt_on_xy[:2]]))
# projected_t_npt_on_xy = np.array([t_npt_on_xy[0], t_npt_on_xy[1], projected_t_npt_z_on_xy[0]])
# projected_point = plane_rotmat.dot(projected_t_npt_on_xy) + plane_center
surface_gm = surface.get_gometricmodel([[-50, 50], [-50, 50]], rgba=[.5, .7, 1, 1])
surface_gm.setpos(plane_center)
surface_gm.setrotmat(plane_rotmat)
surface_gm.reparentTo(base.render)
base.run()