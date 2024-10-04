from wrs import modeling as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    # mt.convert_to_stl("base_link.stl", "base_link.stl", scale_ratio=.001)
    mt.convert_to_stl("inward_left_finger_link.stl", "inward_left_finger_link.stl", scale_ratio=.001)
    mt.convert_to_stl("inward_right_finger_link.stl", "inward_right_finger_link.stl", scale_ratio=.001)