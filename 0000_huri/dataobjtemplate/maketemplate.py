import os
import robothelper
import pickle
import environment.collisionmodel as cm

if __name__ == '__main__':
    yhx = robothelper.RobotHelperX(usereal=False)

    # tubestand
    # tscm = cm.CollisionModel(os.path.join(yhx.root,  "objects",  "tubestand.stl"))
    # tspcd, _ = tscm.samplesurface(radius=2)
    # tspcd = tspcd[tspcd[:,2]>55]
    #
    # pickle.dump(tspcd, open(os.path.join(yhx.root, "dataobjtemplate", "tubestandtemplatepcd.pkl"), "wb"))
    # base = yhx.startworld()
    # pcdcm = cm.CollisionModel(tspcd)
    # pcdcm.reparentTo(base.render)
    # base.run()

    # tubestand_light
    vhcm = cm.CollisionModel(os.path.join(yhx.root,  "objects",  "vacuumhead.stl"))
    vhpcd, _ = vhcm.samplesurface(radius=2)
    vhpcd = vhpcd[vhpcd[:,1]<-5]
    #
    pickle.dump(vhpcd, open(os.path.join(yhx.root, "dataobjtemplate", "vacuumhead_templatepcd.pkl"), "wb"))
    pcdcm = cm.CollisionModel(vhpcd)
    pcdcm.reparentTo(yhx.base.render)
    yhx.base.run()


    # tubestand_light
    tscm = cm.CollisionModel(os.path.join(yhx.root,  "objects",  "tubestand_light.stl"))
    tspcd, _ = tscm.samplesurface(radius=2)
    tspcd = tspcd[tspcd[:,2]>55]
    #
    pickle.dump(tspcd, open(os.path.join(yhx.root, "dataobjtemplate", "tubestand_light_templatepcd.pkl"), "wb"))
    pcdcm = cm.CollisionModel(tspcd)
    pcdcm.reparentTo(yhx.base.render)
    yhx.base.run()

    # handpalm
    # tscm = cm.CollisionModel(os.path.join(yhx.root,  "objects", "handpalm.stl"))
    # tspcd, _ = tscm.samplesurface(radius=2)
    # tspcd = tspcd[tspcd[:,1]<-5]
    # tspcd = tspcd[tspcd[:,2]>5]
    #
    # pickle.dump(tspcd, open(os.path.join(yhx.root, "dataobjtemplate", "handpalmtemplatepcd.pkl"), "wb"))
    # base = yhx.startworld()
    # pcdcm = cm.CollisionModel(tspcd)
    # pcdcm.reparentTo(base.render)
    # base.run()

    # handfinger
    # tscm = cm.CollisionModel(os.path.join(yhx.root,  "objects", "handfinger.stl"))
    # tspcd, _ = tscm.samplesurface(radius=2)
    # tspcd = tspcd[tspcd[:,1]<-5]
    # tspcd = tspcd[tspcd[:,2]>5]
    #
    # pickle.dump(tspcd, open(os.path.join(yhx.root, "dataobjtemplate", "handfingertemplatepcd.pkl"), "wb"))
    # base = yhx.startworld()
    # pcdcm = cm.CollisionModel(tspcd)
    # pcdcm.reparentTo(base.render)
    # base.run()