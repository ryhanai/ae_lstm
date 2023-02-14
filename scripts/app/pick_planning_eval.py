from pick_planning import *
from sim import run_sim_basket_filling as s
import numpy as np
import scipy.linalg
import quaternion
from core.utils import *


def restore_scene(scene_number):
    """
    restore the specified scene in the simulator

    Args:
        n (integer): number of the test scene)

    Returns:
        None

    """

    s.p.setRealTimeSimulation(False)
    s.clear_scene()
    bs = test_data[2][scene_number]
    for object_name, pose in bs:
        oc = s.object_cache.get(object_name)
        body = oc.get_body()
        s.set_pose(body, pose)
        oc.activate()


def move_with_screw(object_name, screw1, v2):
    v1, w1 = screw1

    s.p.setRealTimeSimulation(False)
    body = s.object_cache.get(object_name).get_body()
    s.p.changeDynamics(body, -1, mass=0.)

    init_p, init_q = s.get_pose(body)
    init_p = np.array(init_p)
    init_q = np.array(init_q)
    phase = 1

    history = []
    counter = 0
    
    while True:
        p, q = s.get_pose(body)
        if p[2] >= 0.95:
            for i in range(40):
                goal_p = p + v1 / 240
                s.set_pose(body, (goal_p, init_q))
                s.p.stepSimulation()
            s.p.changeDynamics(body, -1, mass=s.objects[object_name])
            return history
        if phase == 1:
            if scipy.linalg.norm(init_p - p) > 0.05:
                print('moved specified distance -> phase 2')
                phase = 2
            else:
                s.p.performCollisionDetection()
                cps = s.p.getContactPoints(body)
                for cp in cps:
                    body1, body2 = sorted((cp[1], cp[2]))
                    if body1 == s.env.target and body2 == body:
                        if cp[5][2] > 0.75:
                            # print('contact with basket detected -> phase 2')
                            counter += 1
                            if counter > 4:
                                print('contact with basket detected -> phase 2')
                                # init_q = q
                                phase = 2

            goal_pose = (p + v1 / 240, init_q)
            # q0 = quaternion.from_float_array([1, 0, 0, 0])
            # q1 = quaternion.from_float_array([w1[3], w1[0], w1[1], w1[2]])
            # dq = quaternion.slerp(q0, q1, 0, 1, 1/240.)
            # dq = np.array([dq.x, dq.y, dq.z, dq.w])
            # goal_pose = s.env.multiplyTransforms((p, q), (v1 / 240, dq))
        else:  # phase 2
            goal_pose = (p + v2 / 240, init_q)

        s.set_pose(body, goal_pose)
        s.p.stepSimulation()
        history.append(get_object_poses(object_name))
        counter += 1


def get_object_poses(target_object):
    poses = []
    for name, o in s.object_cache.items():
        if name != target_object and o.isActive():
            poses.append(s.get_pose(o.get_body()))
    return poses


def angle_distance(q1, q2):
    return np.arccos(np.clip(np.dot(q1, q2), -1, 1))


def eval_disturbance(poses1, poses2):
    lin_dist = 0
    angle_dist = 0
    for pose1, pose2 in zip(poses1, poses2):
        lin_dist += scipy.linalg.norm(np.array(pose1[0]) - np.array(pose2[0]))
        angle_dist += angle_distance(pose1[1], pose2[1])
    return lin_dist, angle_dist


def analyze(history):
    h = zip(*history)
    max_v = 0
    max_av = 0
    max_displacement = 0
    max_angular_displacement = 0
    for b in h:
        v = np.max([scipy.linalg.norm(np.array(b[i][0]) - b[i+1][0]) for i in range(len(b) - 1)])
        if v > max_v:
            max_v = v
        av = np.max([angle_distance(b[i][1], b[i+1][1]) for i in range(len(b) - 1)])
        if av > max_av:
            max_av = av
        # disp = scipy.linalg.norm(np.array(b[0][0]) - b[-1][0])
        disp = np.max([scipy.linalg.norm(np.array(b[i][0]) - b[0][0]) for i in range(len(b) - 1)])
        if disp > max_displacement:
            max_displacement = disp
        adisp = np.max([angle_distance(b[i][1], b[0][1]) for i in range(len(b) - 1)])
        if adisp > max_angular_displacement:
            max_angular_displacement = adisp
    return max_v * 240, max_displacement, max_av * 240, max_angular_displacement


def do_pick(scene_number, object_name, v=0.3, algorithm='fmap'):
    c = s.get_object_center(object_name)
    print('object center = ', c)
    poses_before_pick = get_object_poses(object_name)
    if algorithm == 'fmap':
        result = pick_direction_plan_sim(scene_number, object_center=c, object_radius=0.05, alpha=0.1)
        pick_direction = result[2]
        pick_rot_axis = result[3]
        pick_rot_omega = result[4]
        pick_v = v * pick_direction / scipy.linalg.norm(pick_direction)
        dq = quaternion.from_rotation_vector(pick_rot_omega * v * pick_rot_axis)
        pick_q = np.array([dq.x, dq.y, dq.z, dq.w])
        pick_screw = (pick_v, pick_q)
    elif algorithm == 'up':
        pick_screw = (np.array([0, 0, v]), [0, 0, 0, 1])

    pick_v2 = np.array([0, 0, v])
    history = move_with_screw(object_name, pick_screw, pick_v2)
    poses_after_pick = get_object_poses(object_name)
    cost = eval_disturbance(poses_before_pick, poses_after_pick)
    print(cost)
    return history


def test(scene_number, target_name):
    restore_scene(scene_number)
    history_fmap = do_pick(scene_number, target_name, algorithm='fmap')
    restore_scene(scene_number)
    history_up = do_pick(scene_number, target_name, algorithm='up')
    
    mvf, mdf, mavf, madf = analyze(history_fmap)
    message('FMAP: max linear vel = {:.5f}[m/s], max linear disp = {:.5f}[m], max angular vel = {:.5f}[rad/s], max angular disp = {:.5f}[rad/s]'.format(mvf, mdf, mavf, madf))
    mvu, mdu, mavu, madu = analyze(history_up)
    message('UP: max linear vel = {:.5f}[m/s], max linear disp = {:.5f}[m], max angular vel = {:.5f}[rad/s], max angular disp = {:.5f}[rad/s]'.format(mvu, mdu, mavu, madu))


s.load_objects()


tests = [
    (62, '007_tuna_fish_can'),
    (62, '008_pudding_box'),
    (63, '008_pudding_box'),
    (72, '011_banana'),
    (76, '016_pear'),
    (87, '008_pudding_box'),
    (90, '011_banana'),
    (91, '011_banana'),
    (117, '008_pudding_box'),
    (122, '061_foam_brick'),
    (122, '017_orange'),
    (122, '056_tennis_ball'),
]

# coords of tuna_fish_can, foam_brick is strange
# some objects penetrate the basket
