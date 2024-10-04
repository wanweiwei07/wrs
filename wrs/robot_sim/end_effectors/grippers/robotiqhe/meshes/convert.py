from wrs import modeling as mt

if __name__ == '__main__':
    '''
    author: weiwei
    date: 20201212osaka
    '''
    mt.convert_to_stl("base.stl", "base_cvt.stl", scale_ratio=.001)
    mt.convert_to_stl("finger1.stl", "finger1_cvt.stl", scale_ratio=.001)
    mt.convert_to_stl("finger2.stl", "finger2_cvt.stl", scale_ratio=.001)
    mt.convert_to_stl("finger1_big.stl", "finger1_big_cvt.stl", scale_ratio=.001)
    mt.convert_to_stl("finger2_big.stl", "finger2_big_cvt.stl", scale_ratio=.001)