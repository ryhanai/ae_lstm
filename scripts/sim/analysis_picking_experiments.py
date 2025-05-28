from pathlib import Path
import pandas as pd
import numpy as np
import re
from sim.generated_grasps import transform_to_centroid


def get_trajectory(data_dir, to_centroid=True, filter_size=1):
    def get_name(prim_name):
        return prim_name.split('/')[-1][1:]

    dt = 1. / 60
    frameNo = 0
    p = Path(data_dir)
    bin_state = pd.read_pickle(p / f'bin_state{frameNo:05d}.pkl')
    if to_centroid:
        bin_state = [(name, transform_to_centroid(name, pos, quat)) for name, (pos, quat) in bin_state]
    trajs = dict([(name, {'pos': [pos], 
                          'quat': [quat], 
                          'v_pos': [np.zeros(3, dtype=np.float32)],
                          'contact_force': [0.0],
                          }) for name, (pos, quat) in bin_state])
    while True:
        frameNo += 1

        try:
            bin_state = pd.read_pickle(p / f'bin_state{frameNo:05d}.pkl')        
        except FileNotFoundError:
            break

        if to_centroid:
            bin_state = [(name, transform_to_centroid(name, pos, quat)) for name, (pos, quat) in bin_state]

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

    # for name, traj in trajs.items():
    #     a = traj['v_pos']
    #     pre_padding = np.tile(a[0], (filter_size, 1))
    #     post_padding = np.tile(a[-1], (filter_size, 1))
    #     a = np.concatenate([pre_padding, a, post_padding])
    #     a = np.array([np.average(a[i-filter_size:i+filter_size+1], axis=0) for i in range(filter_size, len(a) - filter_size)])
    #     traj['v_pos'] = a

    #     a = traj['pos']
    #     padding = np.zeros((filter_size, 3))
    #     a = np.concatenate([padding, a, padding])
    #     a = np.array([np.average(a[i-filter_size:i+filter_size+1], axis=0) for i in range(filter_size, len(a) - filter_size)])
    #     traj['pos'] = a

    return trajs


def eval_trajectories(data_dir, skip_lifting_target=True, summarize=True):
    lifting_target = re.findall('__(.*)__', str(data_dir))[0]
    results = {}

    trajs = get_trajectory(data_dir)

    for name, traj in trajs.items():
        if not (skip_lifting_target and name == lifting_target):
            poss = np.array(traj['pos'])
            total_distance_travelled = np.sum(np.linalg.norm(poss[1:] - poss[:-1], axis=1))
            vels = traj['v_pos']
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


blacklist = [
    '264__052_extra_large_clamp',
    '364__gogotea_straight',
    '580__033_spatula',
    '864__gogotea_straight',
    '908__toppo',
]

def do_evaluation(root_dir='picking_experiment_results', use_blacklist=True):
    root_dir = Path(root_dir)
    results = []
    for data_dir in root_dir.iterdir():
        problem = re.findall('/(.*)_\d\d\d', str(data_dir))[0]
        if use_blacklist and (problem in blacklist):
            print(f'Skipping {data_dir} due to blacklist.')
            continue

        r = eval_trajectories(data_dir)
        print(f' {data_dir}: {r}')
        results.append(r)

    results = np.array(results)
    means = results.mean(axis=0)
    variances = results.var(axis=0)    
    print(f'[MEAN] total distance: {means[0]}, max velocity: {means[1]}, max contact force: {means[2]}')
    print(f'[VAR] total distance: {variances[0]}, max velocity: {variances[1]}, max contact force: {variances[2]}')
    return results


def read_episode_ids(root_dir='picking_experiment_results_UP'):
    root_dir = Path(root_dir)
    episode_ids = []
    for data_dir in root_dir.iterdir():
        episode_ids.append(str(data_dir).split('/')[-1].split('__UP')[0])
    return episode_ids


methods = ['UP', 'GAFS_f0.030_g0.010', 'GAFS_f0.060_g0.010', 'IFS_f0.015', 'IFS_f0.005']

def compare_for_FRONTIERS():
    results = {}
    results['episode_ids'] = read_episode_ids('picking_experiment_results_UP')
    for method in methods:
        results[method] = do_evaluation(f'picking_experiment_results_{method}')
    pd.to_pickle(results, 'picking_experiment_results.pkl')


import csv

def output_to_csv(metric='total_displacement'):
    metric_id = {
        'total_displacement': 0,
        'max_velocity': 1,
        'max_contact_force': 2
    }[metric]

    results = pd.read_pickle('picking_experiment_results.pkl')
    fieldnames = ['episode_id'] + methods

    with open(f'picking_experiments_{metric}.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for vs in zip(results['episode_ids'],
                      results[methods[0]][:,metric_id],
                      results[methods[1]][:,metric_id],
                      results[methods[2]][:,metric_id],                       
                      results[methods[3]][:,metric_id],
                      results[methods[4]][:,metric_id],                      
                      ):
            writer.writerow({'episode_id': vs[0],
                             methods[0]: vs[1],
                             methods[1]: vs[2],
                             methods[2]: vs[3],
                             methods[3]: vs[4],                             
                             methods[4]: vs[5],
                            })