from pathlib import Path
from sre_constants import SUCCESS
import pandas as pd
import numpy as np
import re
import csv

from sympy import root
from sim.generated_grasps import transform_to_centroid
from fmdev.eipl_print_func import print_info


def get_trajectory(data_dir, to_centroid=True, compute_force_from_impulse=False, filter_size=1):
    def get_name(prim_name):
        name = prim_name.split('/')[-1]
        if name != 'table_surface':
            name = name[1:]
        return name

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

        contact_state = pd.read_pickle(p / f'contact_raw_data{frameNo:05d}.pkl')                
        contact_positions, impulse_values, contacting_objects, contact_normals, forces = contact_state

        if compute_force_from_impulse:
            for (prim_name1, prim_name2), f in zip(contacting_objects, impulse_values):
                try:
                    name1 = get_name(prim_name1)
                    name2 = get_name(prim_name2)
                    if name1 != 'table_surface':
                        if name2 == 'table_surface':
                            trajs[name1]['contact_force'][-1] += f
                        else:
                            trajs[name1]['contact_force'][-1] += f / 2.0                            
                    if name2 != 'table_surface':
                        if name1:
                            trajs[name2]['contact_force'][-1] += f
                        else:
                            trajs[name2]['contact_force'][-1] += f / 2.0
                except:
                    pass
        else:
            trajs[name]['contact_force'][-1] = forces[name]

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


    ## Remove objects dropped from the table
    trajs = {k: v for k, v in trajs.items() if np.any(np.array(v['pos'])[:, 2] > 0.65)}

    return trajs


def eval_trajectory(data_dir, summarize=True, success_threshold=0.05):
    problem = re.findall('/(.*)_\d\d\d', str(data_dir))[0]
    scene_number, lifting_target = problem.split('__')
    results = {}
    success = False

    trajs = get_trajectory(data_dir)

    for name, traj in trajs.items():
        if name == lifting_target:
            success = traj['pos'][-1][2] - traj['pos'][0][2] >= success_threshold
            if not success:
                print_info(f"Lifting height {traj['pos'][-1][2] - traj['pos'][0][2]} is not enough")
        else:
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
        return [np.sum(r[:, 0]), np.max(r[:, 1]), np.max(r[:, 2])], success
    else:
        return results, success


blacklist = [
    '264__052_extra_large_clamp',
    '364__gogotea_straight',
    '580__033_spatula',
    '864__gogotea_straight',
    '908__toppo',
]

def do_evaluation(root_dir='picking_experiment_results', use_blacklist=False):
    root_dir = Path(root_dir)
    results = []
    success_flags = []

    for data_dir in root_dir.iterdir():
        problem = re.findall('/(.*)_\d\d\d', str(data_dir))[0]
        if use_blacklist and (problem in blacklist):
            print(f'Skipping {data_dir} due to blacklist.')
            continue

        r, success = eval_trajectory(data_dir)
        print(f' {data_dir}: {r}, success: {success}')
        results.append(r)
        success_flags.append(success)

    results = np.array(results)
    means = results.mean(axis=0)
    stds = results.std(axis=0)    
    print(f'[MEAN] total distance: {means[0]}, max velocity: {means[1]}, max contact force: {means[2]}')
    print(f'[STD] total distance: {stds[0]}, max velocity: {stds[1]}, max contact force: {stds[2]}')
    return results, success_flags


def read_episode_ids(dir):
    episode_ids = []
    for data_dir in dir.iterdir():
        episode_ids.append(str(data_dir).split('/')[-1].split('__UP')[0])
    return episode_ids


methods = ['UP', 'GAFS_f0.030_g0.010', 'GAFS_f0.060_g0.010', 'IFS_f0.015', 'IFS_f0.005']

def compare_for_FRONTIERS(root_dir='picking_experiment_results'):
    root_dir = Path(root_dir)
    results = {}
    results['episode_ids'] = read_episode_ids(root_dir / f'picking_experiment_results_{methods[0]}')    
    for method in methods:
        results[method] = do_evaluation(str(root_dir / f'picking_experiment_results_{method}'))
    pd.to_pickle(results, root_dir / 'picking_experiment_results.pkl')


from functools import reduce

def output_latex_table(picking_result_file='picking_experiment_results.pkl', caption='Disturbance caused by lifting'):
    """
    Output the results of the picking experiments to a LaTeX table to compare different methods for each episode visually.
    """
    
    def compute_success_statistics(success_flags):
        return f"{success_flags.count(True) / len(success_flags):.3f}"

    def format_tag(tag):
        abc = tag.split('_')
        if len(abc) == 3:
            a, b, c = abc
            return f'{a}$(\sigma_f={b[1:]},\sigma_g={c[1:]})$' 
        elif len(abc) == 2:
            a, b = abc
            return f'{a}$(\sigma_f={b[1:]})$'
        else:
            return tag

    results = pd.read_pickle(picking_result_file)

    print('\\begin{table*}[ht]')
    print('\\rowcolors{2}{white}{Gainsboro}')
    print('\\centering')
    print('\\begin{tabular}{ l|llll }')
    print('\\toprule')
    print('\\textbf{Smoothing Method} & \\textbf{Total distance[m]$\\downarrow$} & \\textbf{Max velocity[m/s]$\\downarrow$} & \\textbf{Max contact force[N]$\\downarrow$} & \\textbf{Success rate$\\uparrow$} \\\\ \midrule')

    for k, v in results.items():
        if k == 'episode_ids':
            continue
        
        metrics, success_flags = v
        line = f"{format_tag(k)} & " + \
            reduce(lambda x, y: f"{x} & {y}", 
                    [f"${x[0]:.4f} \pm {x[1]:.4f}$" for x in zip(metrics.mean(axis=0), metrics.std(axis=0))]) + \
            ' & ' + f"${compute_success_statistics(success_flags)}$" + ' \\\\'
        print(line)

    print('\\bottomrule')
    print('\\end{tabular}')
    print(f'\\caption{{\\textbf{{{caption}}}}}')
    print('\\label{tab:prediction_losses}')
    print('\\end{table*}')

def output_to_csv(metric='total_displacement'):
    """
    Output the results of the picking experiments to a CSV file to compare different methods for each episode visually.
    :param metric: The metric to output. Options are 'total_displacement', 'max_velocity', 'max_contact_force'.
    """
    
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


import glob
from PIL import Image

def create_animation_gif_for_episode(episode_path):
                                     
    """
    Create an animation GIF for a specific episode.
    """

    # mdir = Path(root_dir) / f'picking_experiment_results_{method}'
    # scene_id, lifting_target, grasp_number = episode_id
    # episode_dir = mdir / f'{scene_id}__{lifting_target}_{grasp_number:03d}__{method}'

    episode_path = Path(episode_path)
    image_files = sorted(glob.glob(str(episode_path / 'rgb*')))

    images = [Image.open(img) for img in image_files[::2]]
    output_path = episode_path / f"preview__{re.findall('/(.*)__', str(episode_path))[0]}.gif"
    images[0].save(output_path,
                   save_all=True,
                   append_images=images[1:],
                   duration=10,  # 10ms per frame
                   loop=0,  # Loop forever
    )

def create_animation_gifs_for_all_episodes(root_dir='picking_experiment_results_GAFS_f0.030_g0.010'):
    """
    Create animation GIFs for all episodes in the specified root directory.
    """
    root_dir = Path(root_dir)
    for episode_path in root_dir.iterdir():
        if episode_path.is_dir():
            create_animation_gif_for_episode(episode_path)
            print(f'Created GIF for {episode_path}')