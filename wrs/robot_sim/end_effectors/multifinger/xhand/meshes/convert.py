import os
from wrs import mt, rm

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(".stl"):  # 忽略大小写
                mt.convert_to_stl(file, file[:-4]+"_.stl", rotmat=rm.rotmat_from_euler(rm.pi, 0,0))
    # # mt.convert_to_stl("base_link.stl", "base_link.stl", scale_ratio=.001)
    # mt.convert_to_stl("inward_left_finger_link.stl", "inward_left_finger_link.stl", scale_ratio=.001)
    # mt.convert_to_stl("inward_right_finger_link.stl", "inward_right_finger_link.stl", scale_ratio=.001)