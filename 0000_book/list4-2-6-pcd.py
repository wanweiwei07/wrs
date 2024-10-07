import time
from wrs import wd, rm, mcm, mgm

if __name__ == '__main__':
    base = wd.World(cam_pos=rm.np.array([.7, .05, .3]), lookat_pos=rm.np.zeros(3))
    # ウサギのモデルのファイルを用いてCollisionModelを初期化します
    # ウサギ1~5はこのCollisionModelのコピーとして定義します
    objcm_ref = mcm.CollisionModel(initor="./objects/bunnysim.stl",
                                   cdprim_type=mcm.const.CDPrimType.BOX,
                                   cdmesh_type=mcm.const.CDMeshType.DEFAULT)
    objcm_ref.rgba=rm.vec(.9, .75, .35, 1)
    objcm1 = objcm_ref.copy()
    objcm1.pos = rm.np.array([0, -.01, 0])
    objcm1.attach_to(base)
    objcm1.show_cdprim()
    objcm2 = objcm_ref.copy()
    objcm2.pos = rm.np.array([0, .03, 0])
    objcm2.attach_to(base)
    objcm2.show_cdprim()
    objcm2.change_cdprim_type(cdprim_type=mcm.CDPType.CYLINDER)
    tic = time.time()
    result, contact_points = objcm1.is_pcdwith(objcm2, toggle_contacts=True)
    toc = time.time()
    print(f"Time cost is: {toc - tic}", result)
    if result:
        for point in contact_points:
            mgm.gen_sphere(point).attach_to(base)
    base.run()
