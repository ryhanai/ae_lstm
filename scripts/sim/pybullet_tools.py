import pybullet as p
from collections import namedtuple

def Point(x=0, y=0, z=0):
    return np.array([x, y, z])

def get_pose(body):
    return p.getBasePositionAndOrientation(body)

def get_point(body):
    return get_pose(body)[0]

def get_quat(body):
    return get_pose(body)[1] # [x,y,z,w]

def get_euler(body):
    return euler_from_quat(get_quat(body))

def get_base_values(body):
    return base_values_from_pose(get_pose(body))

def set_pose(body, pose):
    (point, quat) = pose
    p.resetBasePositionAndOrientation(body, point, quat)

def set_point(body, point):
    set_pose(body, (point, get_quat(body)))

def set_quat(body, quat):
    set_pose(body, (get_point(body), quat))

def set_euler(body, euler):
    set_quat(body, quat_from_euler(euler))

def set_position(body, x=None, y=None, z=None):
    # TODO: get_position
    position = list(get_point(body))
    for i, v in enumerate([x, y, z]):
        if v is not None:
            position[i] = v
    set_point(body, position)
    return position

def set_orientation(body, roll=None, pitch=None, yaw=None):
    orientation = list(get_euler(body))
    for i, v in enumerate([roll, pitch, yaw]):
        if v is not None:
            orientation[i] = v
    set_euler(body, orientation)
    return orientation


def quat_from_euler(euler):
    return p.getQuaternionFromEuler(euler) # TODO: extrinsic (static) vs intrinsic (rotating)

def euler_from_quat(quat):
    return p.getEulerFromQuaternion(quat) # rotation around fixed axis

def intrinsic_euler_from_quat(quat):
    #axes = 'sxyz' if static else 'rxyz'
    return euler_from_quaternion(quat, axes='rxyz')

def unit_point():
    return (0., 0., 0.)

def unit_quat():
    return quat_from_euler([0, 0, 0]) # [X,Y,Z,W]

def quat_from_axis_angle(axis, angle): # axis-angle
    #return get_unit_vector(np.append(vec, [angle]))
    return quaternion_about_axis(angle, axis)
    #return np.append(math.sin(angle/2) * get_unit_vector(axis), [math.cos(angle / 2)])

def unit_pose():
    return (unit_point(), unit_quat())

# def multiply_transforms(p1, q1, p2, q2):
#     return p.multiplyTransforms(p1, q1, p2, q2)

def multiply_transforms(tf1, tf2):
    return p.multiplyTransforms(tf1[0], tf1[1], tf2[0], tf2[1])

def invert_transform(tf):
    return p.invertTransform(tf[0], tf[1])

NULL_ID = -1
STATIC_MASS = 0

RGB = namedtuple('RGB', ['red', 'green', 'blue'])
RGBA = namedtuple('RGBA', ['red', 'green', 'blue', 'alpha'])
MAX_RGB = 2**8 - 1

RED = RGBA(1, 0, 0, 1)
GREEN = RGBA(0, 1, 0, 1)
BLUE = RGBA(0, 0, 1, 1)
BLACK = RGBA(0, 0, 0, 1)
WHITE = RGBA(1, 1, 1, 1)
BROWN = RGBA(0.396, 0.263, 0.129, 1)
TAN = RGBA(0.824, 0.706, 0.549, 1)
GREY = RGBA(0.5, 0.5, 0.5, 1)
YELLOW = RGBA(1, 1, 0, 1)
TRANSPARENT = RGBA(0, 0, 0, 0)

def create_collision_shape(geometry, pose=unit_pose()):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    point, quat = pose
    collision_args = {
        'collisionFramePosition': point,
        'collisionFrameOrientation': quat,
        #'physicsClientId': CLIENT,
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if 'length' in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args['height'] = collision_args['length']
        del collision_args['length']
    print('ARGS: ', collision_args)
    return p.createCollisionShape(**collision_args)

def create_visual_shape(geometry, pose=unit_pose(), color=None, specular=None):
    # if (color is None): # or not has_gui():
    #     return NULL_ID
    point, quat = pose
    visual_args = {
        'rgbaColor': color,
        'visualFramePosition': point,
        'visualFrameOrientation': quat,
        #'physicsClientId': CLIENT,
    }
    visual_args.update(geometry)
    if specular is not None:
        visual_args['specularColor'] = specular
    return p.createVisualShape(**visual_args)

def create_shape(geometry, pose=unit_pose(), collision=True, **kwargs):
    collision_id = create_collision_shape(geometry, pose=pose, **kwargs) if collision else NULL_ID
    visual_id = create_visual_shape(geometry, pose=pose, **kwargs) # if collision else NULL_ID
    return collision_id, visual_id

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id, baseVisualShapeIndex=visual_id)

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def get_mesh_geometry(mesh_file, scale, static=False):
    return {
        'shapeType': p.GEOM_MESH,
        'fileName': mesh_file,
        'meshScale': scale,
        'flags': p.GEOM_FORCE_CONCAVE_TRIMESH if static else 0
    }

def create_sphere(radius, mass=STATIC_MASS, color=BLUE, **kwargs):
    collision_id, visual_id = create_shape(get_sphere_geometry(radius), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)

def create_mesh_body(mesh_file, mass=0.1, scale=[1.0,1.0,1.0], **kwargs):
    collision_id, visual_id = create_shape(get_mesh_geometry(mesh_file, scale), **kwargs)
    return create_body(collision_id, visual_id, mass=mass)
