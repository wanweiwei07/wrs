import matplotlib.pyplot as plt
from vision.coded_aperture.code_aperture import mura

result = mura(rank=5)
result.plot()
plt.show()