from wrs import modeling as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201207osaka
    '''
    mt.convert_to_stl("ur3e_dual_base.stl", "ur3e_dual_base_cvt.stl", scale_ratio=.001)