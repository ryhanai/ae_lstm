import numpy as np
import pandas as pd
import genesis as gs

from dataset.object_loader import ObjectInfo

# visopt = gs.options.vis.VisOptions()
# visopt.contact_force_scale = 0.1

########################## 初期化 ##########################
gs.init(backend=gs.gpu)

########################## シーンの作成 ##########################
scene = gs.Scene(
    viewer_options = gs.options.ViewerOptions(
        camera_pos    = (0, -3.5, 2.5),
        camera_lookat = (0.0, 0.0, 0.5),
        camera_fov    = 30,
        res           = (960, 640),
        max_FPS       = 60,
    ),
    sim_options = gs.options.SimOptions(
        dt = 0.01,
    ),
    vis_options = gs.options.VisOptions(
        show_world_frame = False, # `world`の原点座標系を可視化
        world_frame_size = 1.0, # 座標系の長さ（メートル単位）
        show_link_frame  = False, # エンティティリンクの座標系は非表示
        show_cameras     = False, # 追加されたカメラのメッシュと視錐台を非表示
        plane_reflection = True, # 平面反射を有効化
        ambient_light    = (0.2, 0.2, 0.2), # 環境光設定
        contact_force_scale = 0.4, # 接触力のスケール
    ),
    show_viewer = True,
)

########################## エンティティ ##########################
plane = scene.add_entity(
    gs.morphs.Plane(),
)

# Add table
table_surface = scene.add_entity(
    morph=gs.morphs.Box(
        size = (1.0, 1.0, 0.04),
        pos  = (0.0, 0.0, 0.69),
        fixed = True,
    ),
    surface=gs.surfaces.Rough(
        # color = (0.14, 0.9, 0.65, 1.0),
        color = (0.14, 0.8, 0.45, 1.0),        
    )
)

# Add robot
franka = scene.add_entity(
    gs.morphs.MJCF(
        file  = 'xml/franka_emika_panda/panda.xml',
        pos = (-0.6, 0.0, 0.7),
    ),
    visualize_contact=True,
)

## Add cameras


## Load scene data and add procucts
oi = ObjectInfo()

def load_bin_state(scene_idx):
    active_products = []
    bs = pd.read_pickle(f"/home/ryo/Dataset/forcemap/tabletop240304/bin_state{scene_idx:05d}.pkl")
    for name, (pos, ori) in bs:  # quaternion is scholar first: (w, x, y, z)
        ori[0], ori[1], ori[2], ori[3] = ori[3], ori[0], ori[1], ori[2]
        product = scene.add_entity(
            morph=gs.morphs.Mesh(
                file=oi.obj_file(name, with_scale=False),
                pos=pos,
                quat=ori,
            ),
            visualize_contact=True,
        )        

load_bin_state(13)


def get_entity_name(entity):
    return entity.morph.file.split('/')[-2]


## Get Contact Force
def get_contact_forces():
    ret = []
    for entity in scene.entities:
        if isinstance(entity.morph, gs.options.morphs.Mesh):
            contacts = entity.get_contacts()
            ret.append(contacts)
    return ret


# hand: gs.RigidLink = franka.get_link('hand')
# contacts = hand.entity.get_contacts()  # ids are in linkA, linkB
# link_a_ids = contacts['link_a']
# link_b_ids = contacts['link_b']
# collided_link_a = scene.rigid_solver.links[link_a_ids[0]]
# collided_link_b = scene.rigid_solver.links[link_b_ids[0]]
# print(collided_link_a.name, collided_link_b.name)


## Grasp sampler (MIMO)
import transforms3d as tf
import trimesh
from mimo.data_gen.panda_sample import PandaGripper, sample_multiple_grasps

def sample_grasps(object_name, n_grasps=20):
    gripper = PandaGripper()
    obj_mesh = trimesh.load(oi.obj_file(object_name, with_scale=False), force='mesh')
    trans, quality = sample_multiple_grasps(n_grasps, obj_mesh, gripper, systematic_sampling=False)
    quality = np.array(quality["quality_antipodal"])
    trans = trans[quality > 0.05]        
    grasp = trans.tolist()
    # print(grasp)
    print(f"Number of high quality grasps: {len(grasp)}")
    return grasp


lifting_target = '052_extra_large_clamp'
grasps = sample_grasps(lifting_target, n_grasps=50)


########################## ビルド ##########################
scene.build()

jnt_names = [
    'joint1',
    'joint2',
    'joint3',
    'joint4',
    'joint5',
    'joint6',
    'joint7',
    'finger_joint1',
    'finger_joint2',
]
dofs_idx = [franka.get_joint(name).dof_idx_local for name in jnt_names]

############ オプション：制御ゲインの設定 ############
# 位置ゲインの設定
franka.set_dofs_kp(
    kp             = np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]),
    dofs_idx_local = dofs_idx,
)
# 速度ゲインの設定
franka.set_dofs_kv(
    kv             = np.array([450, 450, 350, 350, 200, 200, 200, 10, 10]),
    dofs_idx_local = dofs_idx,
)
# 安全のための力の範囲設定
franka.set_dofs_force_range(
    lower          = np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
    upper          = np.array([ 87,  87,  87,  87,  12,  12,  12,  100,  100]),
    dofs_idx_local = dofs_idx,
)


# Initial pose
franka.set_dofs_position([0, -0.5, 0, -2, 0, np.pi/2, 1, 0, 0,], dofs_idx)
scene.step()



# 'hand'リンクはx周りに180度回転させただけに見える．
# z方向はMIMOと同じ．
# 指のスライド方向が，Genesisはy軸，MIMOはx軸なので，
# 追加でz軸周りにpi/2回せば一致するはず
# tf.euler.quat2euler(q.to('cpu'))
# Out[68]: (-3.126955534659113, -0.06757781592988017, -0.2150900160806781)


## Set grasp pose
target_quat = np.array([0, 1, 0, 0])  # pointing downwards, Genesis　modelではyが上
    center = np.array([0.4, -0.2, 0.25])
    r = 0.1

    ee_link = robot.get_link("hand")

    for i in range(0, 2000):
        target_pos = center + np.array([np.cos(i / 360 * np.pi), np.sin(i / 360 * np.pi), 0]) * r

        target_entity.set_qpos(np.concatenate([target_pos, target_quat]))
        q, err = robot.inverse_kinematics(
            link=ee_link,
            pos=target_pos,
            quat=target_quat,
            return_error=True,
            rot_mask=[False, False, True],  # for demo purpose: only care about direction of z-axis
        )
        print("error:", err)

        # Note that this IK example is only for visualizing the solved q, so here we do not call scene.step(), but only update the state and the visualizer
        # In actual control applications, you should instead use robot.control_dofs_position() and scene.step()
        robot.set_qpos(q)
        scene.visualizer.update()


# ハードリセット
# for i in range(150):
#     if i < 50:
#         franka.set_dofs_position(np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]), dofs_idx)
#     elif i < 100:
#         franka.set_dofs_position(np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]), dofs_idx)
#     else:
#         franka.set_dofs_position(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]), dofs_idx)

#     scene.step()

# # PD制御
# for i in range(1250):
#     if i == 0:
#         franka.control_dofs_position(
#             np.array([1, 1, 0, 0, 0, 0, 0, 0.04, 0.04]),
#             dofs_idx,
#         )
#     elif i == 250:
#         franka.control_dofs_position(
#             np.array([-1, 0.8, 1, -2, 1, 0.5, -0.5, 0.04, 0.04]),
#             dofs_idx,
#         )
#     elif i == 500:
#         franka.control_dofs_position(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     elif i == 750:
#         # 最初の自由度を速度で制御し、残りを位置で制御
#         franka.control_dofs_position(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])[1:],
#             dofs_idx[1:],
#         )
#         franka.control_dofs_velocity(
#             np.array([1.0, 0, 0, 0, 0, 0, 0, 0, 0])[:1],
#             dofs_idx[:1],
#         )
#     elif i == 1000:
#         franka.control_dofs_force(
#             np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
#             dofs_idx,
#         )
#     # これは与えられた制御コマンドに基づいて計算された制御力です
#     # 力制御を使用している場合、これは与えられた制御コマンドと同じです
#     print('control force:', franka.get_dofs_control_force(dofs_idx))

#     # これは自由度が実際に経験している力です 
#     print('internal force:', franka.get_dofs_force(dofs_idx))

#    scene.step()
