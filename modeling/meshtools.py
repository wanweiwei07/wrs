import basis.trimesh as trimesh

def scale(obj, ratio):
    """
    :param obj: trimesh or file path
    :param ratio: float, scale all axis equally
    :return:
    author: weiwei
    date: 20201116
    """
    if isinstance(obj, trimesh.Trimesh):
        tmpmesh = obj.copy()
        tmpmesh.apply_scale(ratio)
        return tmpmesh
    elif isinstance(obj, str):
        originalmesh = trimesh.load_mesh(obj)
        tmpmesh = originalmesh.copy()
        tmpmesh.apply_scale(ratio)
        return tmpmesh


def scale_and_save(obj, ratio, savename):
    """
    :param obj: trimesh or file path
    :param ratio: float, scale all axis equally
    :param savename: filepath+filename
    :return:
    author: weiwei
    date: 20201116
    """
    tmptrimesh = scale(obj, ratio)
    tmptrimesh.export(savename)


if __name__ == '__main__':
    # The following contents are commented out to avoid mis-exec.
    pass
    # root = "./objects/"
    # for subdir, dirs, files in os.walk(root):
    #     for file in files:
    #         print(root+file)
    #         scale_and_save(root+file, .001, file)
    # scale_and_save("./objects/block.stl", .001, "block.stl")
    # scale_and_save("./objects/bowlblock.stl", .001, "bowlblock.stl")