import numpy as np
import basis.trimesh as trm

def scale(obj, scale_ratio):
    """
    :param obj: trimesh or file path
    :param scale_ratio: float, scale all axis equally
    :return:
    author: weiwei
    date: 20201116
    """
    if isinstance(obj, trm.Trimesh):
        tmpmesh = obj.copy()
        tmpmesh.apply_scale(scale_ratio)
        return tmpmesh
    elif isinstance(obj, str):
        originalmesh = trm.load(obj)
        tmpmesh = originalmesh.copy()
        tmpmesh.apply_scale(scale_ratio)
        return tmpmesh


def scale_and_save(obj, scale_ratio, savename):
    """
    :param obj: trimesh or file path
    :param scale_ratio: float, scale all axis equally
    :param savename: filepath+filename
    :return:
    author: weiwei
    date: 20201116
    """
    tmptrimesh = scale(obj, scale_ratio)
    tmptrimesh.export(savename)

def convert_to_stl(obj, savename, scale_ratio=1, pos=np.zeros(3), rotmat=np.eye(3)):
    """
    :param obj: trimesh or file path
    :param savename:
    :return:
    author: weiwei
    date: 20201207
    """
    trimesh = trm.load(obj)
    tmptrimesh = scale(trimesh, scale_ratio)
    tmptrimesh.vertices = rotmat.dot(tmptrimesh.vertices.T).T
    tmptrimesh.vertcies = tmptrimesh.vertices+pos
    tmptrimesh.face_normals = rotmat.dot(tmptrimesh.face_normals.T).T
    tmptrimesh.export(savename)

if __name__ == '__main__':
    # The following contents are commented out to avoid mis-exec.
    pass
    # root = "./objects/"
    # for subdir, dirs, files in os.walk(root):
    #     for file in files:
    #         print(root+file)
    #         scale_and_save(root+file, .001, file)
    # scale_and_save("./objects/block.meshes", .001, "block.meshes")
    # scale_and_save("./objects/bowlblock.meshes", .001, "bowlblock.meshes")
    convert_to_stl("base.dae", "base.meshes")