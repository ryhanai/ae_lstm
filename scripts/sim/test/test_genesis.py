import torch
import genesis  as gs
from dataset.object_loader import ObjectInfo

oi = ObjectInfo()

########################### Initialize ##########################

gs.init(backend=gs.gpu)

scene = gs.Scene(
    vis_options = gs.options.VisOptions(
        show_world_frame = False,   # Show xyz axes
        world_frame_size = 1.0,
        show_link_frame  = False,
        show_cameras     = False,
        plane_reflection = True,
        ambient_light    = (0.1, 0.1, 0.1),
    ),
)

cam = scene.add_camera(
    res    = (200, 200),
    pos    = (0.0, -1.0, 0.7),
    lookat = (0, 0, 0.7),
    fov    = 45,
    GUI    = True,
)

########################### entities ##########################
plane = scene.add_entity(
    gs.morphs.Plane()
)

bunny = scene.add_entity(
    morph=gs.morphs.Mesh(
        file=oi.obj_file('jif')[0],
        pos=(0, 0, 0.7),
        fixed=True,
    )
)

########################## build ##########################

scene.build(n_envs = 1, env_spacing = (2.0, 2.0))
cam.render(depth=True, segmentation=False)

for i in range(1_000):
    scene.step()

    # change camera position
    cam.set_pose(
        pos = (3.0 * torch.sin(torch.deg2rad(torch.tensor(i*30))), 3.0 * torch.cos(torch.deg2rad(torch.tensor(i*30))), 0.7),
        lookat = (0, 0, 0.7),
    )
    img, depth, seg, normal = cam.render(depth=True, segmentation=True, normal=True)