import numpy as np


def sample_light_position():
    return np.array([-1, -1, 2.5]) + np.array([2, 2, 1]) * np.random.random(3)


def sample_shadow_map_intensity():
    return 0.2 + 0.8 * np.random.random() # [0.2,1.0]


def sample_shadow_map_resolution():
    return np.random.choice([2048, 4096, 8192, 16384])


def sample_shadow_map_world_size():
    return np.random.randint(2,11)  # 2 - 10


def sample_camera_intrinsics(camera_config):
    fov = camera_config['fov'] - 1.0 + 2.0 * np.random.random()
    aspect_ratio = camera_config['aspectRatio'] - 0.1 + 0.2 * np.random.random()
    return camera_config['imageSize'], fov, aspect_ratio


def sample_camera_extrinsics(camera_config):
    a = 0.01
    b = 0.02
    eye_position = np.array(camera_config['eyePosition']) - a + b * np.random.random(3)
    target_position = np.array(camera_config['targetPosition']) - a + b * np.random.random(3)
    up_vector = np.array(camera_config['upVector']) - a + b * np.random.random(3)
    return eye_position, target_position, up_vector


def sample_object_color(body_id):
    # rgba = np.array(S.p.getVisualShapeData(body_id)[0][7])
    rgba = np.ones(4)
    return rgba - np.array([0.3, 0.3, 0.3, 0.0]) * np.random.random(4)


def sample_object_specular_color(body_id):
    return 2.0 * np.random.random(3)  # 0 - 10


def sample_scene_parameters(object_cache, camera):
    """
        camera = env.getCamera('camera1')
    """
    config = {}
    config['light_position'] = sample_light_position()
    config['shadow_map_intensity'] = sample_shadow_map_intensity()
    config['shadow_map_resolution'] = sample_shadow_map_resolution()
    config['shadow_map_world_size'] = sample_shadow_map_world_size()

    camera_config = camera.cameraConfig
    config['camera_intrinsics'] = sample_camera_intrinsics(camera_config)
    config['camera_extrinsics'] = sample_camera_extrinsics(camera_config)

    object_configs = []
    for k, v in object_cache.items():
        if v.isActive():
            object_config = {'name': k}
            object_config['color'] = sample_object_color(v.body)  # RGBA
            object_config['specular_color'] = sample_object_specular_color(v.body)  # RGB
            object_configs.append(object_config)
    config['objects'] = object_configs

    return config

        
