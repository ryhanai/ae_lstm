from pathlib import Path
import pandas as pd
import numpy as np
import re


def get_trajectory(data_dir, filter_size=1):
    def get_name(prim_name):
        return prim_name.split('/')[-1][1:]

    dt = 1. / 60
    frameNo = 0
    p = Path(data_dir)
    bin_state = pd.read_pickle(p / f'bin_state{frameNo:05d}.pkl')
    trajs = dict([(name, {'pos': [pose[0]], 
                          'quat': [pose[1]], 
                          'v_pos': [np.zeros(3, dtype=np.float32)],
                          'contact_force': [0.0],
                          }) for name, pose in bin_state])
    while True:
        frameNo += 1

        try:
            bin_state = pd.read_pickle(p / f'bin_state{frameNo:05d}.pkl')        
        except FileNotFoundError:
            break

        for name, (pos, quat) in bin_state:
            last_pos = trajs[name]['pos'][-1]
            trajs[name]['v_pos'].append((pos - last_pos) / dt)
            trajs[name]['pos'].append(pos)
            trajs[name]['quat'].append(quat)
            trajs[name]['contact_force'].append(0.0)

        for traj in trajs:
            contact_state = pd.read_pickle(p / f'contact_raw_data{frameNo:05d}.pkl')                
            contact_positions, impulse_values, contacting_objects, contact_normals = contact_state
            for (prim_name1, prim_name2), f in zip(contacting_objects, impulse_values):
                try:
                    name1 = get_name(prim_name1)
                    name2 = get_name(prim_name2)
                    trajs[name1]['contact_force'][-1] += f
                    trajs[name2]['contact_force'][-1] += f
                except:
                    pass

    for name, traj in trajs.items():
        a = traj['v_pos']
        padding = np.zeros((filter_size, 3))
        a = np.concatenate([padding, a, padding])
        # a = [np.max(a[i-filter_size:i+filter_size+1], axis=0) for i in range(filter_size, len(a) - filter_size)]
        a = [a[i] for i in range(filter_size, len(a) - filter_size)]        
        traj['v_pos_s'] = a

    return trajs


def eval_trajectories(data_dir, skip_lifting_target=True, summarize=True):
    lifting_target = re.findall('__(.*)__', str(data_dir))[0]
    results = {}

    trajs = get_trajectory(data_dir)

    for name, traj in trajs.items():
        if not (skip_lifting_target and name == lifting_target):
            poss = np.array(traj['pos'])
            total_distance_travelled = np.sum(np.linalg.norm(poss[1:] - poss[:-1], axis=1))
            vels = traj['v_pos_s']
            max_velocity = np.max(np.linalg.norm(vels, axis=1))
            cfs = traj['contact_force']
            max_contact_force = np.max(cfs)
            results[name] = {
                'total_distance': total_distance_travelled,
                'max_velocity': max_velocity,
                'max_contact_force': max_contact_force,
                }

    if summarize:
        r = np.array([list(r.values()) for r in results.values()])
        return [np.sum(r[:, 0]), np.max(r[:, 1]), np.max(r[:, 2])]
    else:
        return results
    

def do_evaluation(root_dir='picking_experiment_results'):
    root_dir = Path(root_dir)
    results = []
    for data_dir in root_dir.iterdir():
        r = eval_trajectories(data_dir)
        print(f' {data_dir}: {r}')
        results.append(r)

    results = np.array(results)
    print(f'total distance: {np.average(results[:, 0])}, max velocity: {np.average(results[:, 1])}, max contact force: {np.average(results[:, 2])}')

