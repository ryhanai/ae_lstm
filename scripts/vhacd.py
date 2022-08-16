import pybullet as p
import pybullet_data as pd

p.connect(p.DIRECT)
p.setAdditionalSearchPath("../")
#p.vhacd("../specification/meshes/objects/glass1.obj", "../specification/meshes/objects/glass1_vhacd.obj", "log.txt")
#p.vhacd("../specification/meshes/objects/pitcher1.obj", "../specification/meshes/objects/picher1_vhacd.obj", "log.txt")
#p.vhacd("../specification/meshes/objects/pen_cavity_in_out.obj", "../specification/meshes/objects/pen_cavity_in_out_vhacd.obj", "log.txt", resolution=24000000, concavity=0.0001, gamma=0.0005, maxNumVerticesPerCH=1024, minVolumePerCH=0.01)
p.vhacd("../specification/meshes/objects/pen.obj", "../specification/meshes/objects/pen_vhacd.obj", "log.txt")
