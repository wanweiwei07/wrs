class Lookuptable(object):
    def __init__(self, bins, GradMag, GradDir, GradX, GradY, Zeropoint, Scale, Pixmm, FrameSize):
        self.bins = bins
        self.GradMag = GradMag
        self.GradDir = GradDir
        self.GradX = GradX
        self.GradY = GradY
        self.Zeropoint = Zeropoint
        self.Scale = Scale
        self.Pixmm = Pixmm
        self.FrameSize = FrameSize
