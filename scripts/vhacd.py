import pybullet as p
import pybullet_data as pd

p.connect(p.DIRECT)
p.setAdditionalSearchPath("../")
#p.vhacd("../specification/meshes/objects/glass1.obj", "../specification/meshes/objects/glass1_vhacd.obj", "log.txt")
p.vhacd("../specification/meshes/objects/pitcher1.obj", "../specification/meshes/objects/picher1_vhacd.obj", "log.txt")

