from pandaplotutils import pandactrl as pandactrl
from wrs import manipulation as yi, manipulation as freegrip
import utiltools.robotmath as rm
import numpy as np
import database.dbaccess as db
import environment.collisionmodel as cm

if __name__=="__main__":

    base = pandactrl.World(camp = [300,300,500], lookatpos= [0, 0, 60])
    base.pggen.plotAxis(base.render, length=30, thickness=2)

    objpath = "./objects/tubelarge.stl"
    obj = cm.CollisionModel(objinit=objpath)
    obj.setColor(.8, .6, .3, .5)
    obj.reparentTo(base.render)

    hndfa = yi.YumiIntegratedFactory()
    pregrasps = []

    yihnd = hndfa.genHand()
    c0nvec = np.array([0,-1,0])
    approachvec = np.array([1, 0, 0])
    for z in [60,90]:
        for anglei in [0,22.5,45,67.5,90,112.5,135,157.5]:
            newcv = np.dot(rm.rodrigues((0,0,1), anglei),c0nvec)
            tempav = np.dot(rm.rodrigues((0,0,1), anglei),approachvec)
            for anglej in [0,22.5,45,67.5,90,112.5,135,157.5,180,202.5,225,247.5,270,292.5,315,337.5]:
                newyihand = yihnd.copy()
                newav = np.dot(rm.rodrigues(newcv,anglej), tempav)
                base.pggen.plotAxis(newyihand.hndnp, length=15, thickness=2)
                pregrasps.append(newyihand.approachAt(0,0,z,newcv[0],newcv[1],newcv[2],newav[0],newav[1],newav[2],jawwidth=17))
                pregrasps.append(newyihand.approachAt(0,0,z,-newcv[0],-newcv[1],-newcv[2],-newav[0],-newav[1],newav[2],jawwidth=17))
                # newyihand.setColor(.7,.7,.7,1)
                newyihand.reparentTo(base.render)

    # yihnd = hndfa.genHand(usesuction=True)
    # c0nvec = np.array([0, -1, 0])
    # approachvec = np.array([1, 0, 0])
    # for z in [60, 90]:
    #     for anglei in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]:
    #         newcv = np.dot(rm.rodrigues((0, 0, 1), anglei), c0nvec)
    #         tempav = np.dot(rm.rodrigues((0, 0, 1), anglei), approachvec)
    #         for anglej in [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]:
    #             newyihand = yihnd.copy()
    #             newav = np.dot(rm.rodrigues(newcv, anglej), tempav)
    #             base.pggen.plotAxis(newyihand.hndnp, axis_length=15, major_radius=2)
    #             pregrasps.append(
    #                 newyihand.approachAt(0, 0, z, newcv[0], newcv[1], newcv[2], newav[0], newav[1],
    #                                      newav[2], ee_values=17))

    # # toggle this part for manually defined plans
    fgplanner = freegrip.Freegrip(objpath, yihnd)
    gdb = db.GraspDB(database="yumi")
    print("Saving to database...")
    fgplanner.saveManuallyDefinedToDB(gdb, pregrasps)

    base.run()