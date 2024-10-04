import os
import database.dbaccess as db
import pandaplotutils.pandactrl as pandactrl
from wrs import manipulation as yi, manipulation as freegrip
import environment.collisionmodel as cm

if __name__=='__main__':

    base = pandactrl.World(camp=[2700, -2000, 2000], lookatpos=[0, 0, 500])

    _this_dir, _ = os.path.split(__file__)
    objpath = os.path.join(_this_dir, "objects", "tubelarge.stl")

    hndfa = yi.YumiIntegratedFactory()
    hnd = hndfa.genHand()
    fgplanner = freegrip.Freegrip(objpath, hnd, faceangle = .95, segangle = .95, refine1min=2, refine1max=30,
                 refine2radius=5, fpairparallel=-0.95, hmax=5, objmass = 5.0, bypasssoftfgr = True, togglebcdcdebug = False, useoverlap = True)
    objcm = cm.CollisionModel(objinit = objpath)
    objcm.reparentTo(base.render)
    objcm.setColor(.3,.3,.3,1)

    # toggle this part for automatic planning
    print("Planning...")
    fgplanner.planGrasps()
    gdb = db.GraspDB(database="yumi")
    print("Saving to database...")
    fgplanner.saveToDB(gdb)

    print("Loading from database...")
    fgdata = freegrip.FreegripDB(gdb, objcm.name, hnd.name)
    print("Plotting...")
    fgdata.plotHands(base.render, hndfa, rgba=(0,1,0,.3))
    base.run()