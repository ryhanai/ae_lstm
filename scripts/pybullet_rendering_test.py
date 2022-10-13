import pybullet as pb
from pybullet_rendering import RenderingPlugin
from pybullet_rendering.render.panda3d import P3dRenderer # panda3d-based renderer
from pybullet_rendering.render.pyrender import PyrRenderer # pyrender-based renderer

client_id = pb.connect(pb.DIRECT)

# bind your renderer to pybullet
renderer = P3dRenderer(multisamples=4) # or PyrRenderer(platform='egl', egl_device=1)
plugin = RenderingPlugin(client_id, renderer)

# render thru the standard pybullet API
w = 320
h = 240
projectionMatrix = pb.computeProjectionMatrixFOV(50, w/h, 0.1, 2.0)
viewMatrix = pb.computeViewMatrix([0,-0.95,1.1], [0,-0.45,0.6], [0,0,1])
w, h, rgba, depth, mask = pb.getCameraImage(w, h, projectionMatrix=projectionMatrix, viewMatrix=viewMatrix)
