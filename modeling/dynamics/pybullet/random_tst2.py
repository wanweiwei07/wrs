import pybullet as p
import time

p_client = p.connect(p.DIRECT)
p.setGravity(0,0,-9.81)

for i in range (10000):
    p.stepSimulation()
    time.sleep(1./240.)
