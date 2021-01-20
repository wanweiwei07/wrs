import time
import numpy as np

a = np.eye(1000)

tic = time.time()
b = np.vstack([a]*100)
toc = time.time()
print(toc-tic)

tic = time.time()
c = np.array([a.tolist()]*100)
toc = time.time()
print(toc-tic)

print(np.where(a==2))