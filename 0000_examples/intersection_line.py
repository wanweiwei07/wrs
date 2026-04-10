import numpy as np
import pyvista as pv
from wrs import wd, rm, mcm, mgm


def trimesh2pv(objcm):
    m = objcm.trm_mesh
    verts = m.vertices
    R = objcm.rotmat  # (3,3)
    t = objcm.pos  # (3,)
    verts = (R @ verts.T).T + t
    faces = m.faces
    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int64)
    return pv.PolyData(verts, faces_pv)

def pvpoly_to_segs(poly):
    pts = poly.points          # (M,3)
    cells = poly.lines         # flat array
    segs = []
    i = 0
    while i < len(cells):
        n = cells[i]           # number of points in this polyline
        ids = cells[i+1:i+1+n] # indices of these points
        i += n + 1
        for a, b in zip(ids[:-1], ids[1:]):
            segs.append([pts[a], pts[b]])
    return np.array(segs)

if __name__ == '__main__':
    
    base = wd.World(cam_pos=rm.np.array([.7, .05, .3]), lookat_pos=rm.np.zeros(3))
    # ウサギのモデルのファイルを用いてCollisionModelを初期化します
    # ウサギ1~5はこのCollisionModelのコピーとして定義します
    object_ref = mcm.CollisionModel(initor="./objects/bunnysim.stl",
                                    cdmesh_type=mcm.const.CDMeshType.DEFAULT,
                                    cdprim_type=mcm.const.CDPrimType.AABB)
    object_ref.rgba = rm.np.array([.9, .75, .35, 1])

    object1 = object_ref.copy()
    object1.pos = rm.np.array([0, -0.02, 0])
    object1.alpha=.3

    object2 = object_ref.copy()
    object2.pos = rm.np.array([0, 0.02, 0])
    object2.alpha=.3

    object1.attach_to(base)
    object2.attach_to(base)

    pv1 = trimesh2pv(object1)
    pv2 = trimesh2pv(object2)
    lines, _, _ = pv1.intersection(pv2,
                                   split_first=True,
                                   split_second=False)
    print(lines.points.shape)
    mgm.gen_linesegs(linesegs=pvpoly_to_segs(lines), thickness=0.002, rgb=[1, 0, 0]).attach_to(base)

    base.run()
