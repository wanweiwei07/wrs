import modeling.meshtools as mt

if __name__ == '__main__':
    '''
    The Universal Robot UR3E dae file is not well developed.
    There is an inherent transformation.
    Add the following transformation to dae.py loader to bypass the inherent transformation and convert them to stl.
    author: weiwei
    date: 20201207osaka
    # vertices[:, [1, 2]] = vertices[:, [2, 1]]
    # vertices[:, 1] = -vertices[:, 1]
    # face_normals[:, [1, 2]] = face_normals[:, [2, 1]]
    # face_normals[:, 1] = -face_normals[:, 1]
    '''
    mt.convert_to_stl("base.dae", "base.stl")
    mt.convert_to_stl("shoulder.dae", "shoulder.stl")
    mt.convert_to_stl("upperarm.dae", "upperarm.stl")
    mt.convert_to_stl("forearm.dae", "forearm.stl")
    mt.convert_to_stl("wrist1.dae", "wrist1.stl")
    mt.convert_to_stl("wrist2.dae", "wrist2.stl")
    mt.convert_to_stl("wrist3.dae", "wrist3.stl")