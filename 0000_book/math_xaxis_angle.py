from wrs import rm

if __name__ == '__main__':
    x_ax = rm.np.array([1,0,0])
    x30_rotmat = rm.rotmat_from_axangle(x_ax, rm.radians(30))
    print(x30_rotmat)