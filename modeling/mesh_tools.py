import numpy as np
import basis.trimesh as trm
import basis.robot_math as rm

# def scale(obj, scale_ratio):
#     """
#     DEPRECATED: scale is no long supported 20230821
#     :param obj: trimesh or file path
#     :param scale_ratio: float, scale all axis equally
#     :return:
#     author: weiwei
#     date: 20201116
#     """
#     if isinstance(obj, trm.Trimesh):
#         tmpmesh = obj.copy()
#         tmpmesh.apply_scale(scale_ratio)
#         return tmpmesh
#     elif isinstance(obj, str):
#         originalmesh = trm.load(obj)
#         tmpmesh = originalmesh.copy()
#         tmpmesh.apply_scale(scale_ratio)
#         return tmpmesh
#
#
# def scale_and_save(obj, scale_ratio, save_name):
#     """
#     DEPRECATED: scale is no long supported 20230821
#     :param obj: trimesh or file path
#     :param scale_ratio: float, scale all axis equally
#     :param save_name: filepath+filename
#     :return:
#     author: weiwei
#     date: 20201116
#     """
#     tmptrimesh = scale(obj, scale_ratio)
#     tmptrimesh.export(save_name)
#
# def convert_to_stl(obj, save_name, scale_ratio=1, pos=np.zeros(3), rotmat=np.eye(3)):
#     """
#     DEPRECATED: scale is no long supported 20230821
#     :param obj: trimesh or file path
#     :param save_name:
#     :return:
#     author: weiwei
#     date: 20201207
#     """
#     trimesh = trm.load(obj)
#     if type(scale_ratio) is not list:
#         scale_ratio = [scale_ratio, scale_ratio, scale_ratio]
#     tmptrimesh = scale(trimesh, scale_ratio)
#     pos = rm.homomat_from_posrot(pos, rotmat)
#     tmptrimesh.apply_transform(pos)
#     tmptrimesh.export(save_name)

def convert_to_stl(obj, save_name, pos=np.zeros(3), rotmat=np.eye(3)):
    """
    :param obj: trimesh or file path
    :param save_name:
    :return:
    author: weiwei
    date: 20201207
    """
    trm_model = trm.load(obj)
    homomat = rm.homomat_from_posrot(pos, rotmat)
    trm_model.apply_transform(homomat)
    trm_model.export(save_name)

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