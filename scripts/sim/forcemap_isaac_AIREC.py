# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--method',
                    choices=['sim', 'smoothing'], 
                    help='sim: generate dataset using Isaac Sim \
                        | smoothing: apply SDF'
                    )
parser.add_argument('--num_scenes', type=int, default=2,
                    help='number of scenes generated by "sim"')

args = parser.parse_args()


if args.method == 'sim':
    env_config = {
        'headless': True,
        'asset_root': os.environ["HOME"] + "/Dataset/scenes",
        'scene_usd': "green_table_scene.usd",
        'viewpoint': ([-0.42, 0, 1.15], [0, 45, 0]),  # euler angles in degree
        'viewpoint_randomization_range': (0.02, 3),  # ([m], [degree])
        'object_set': "ycb_conveni_v1",
        'crop_size': [768, 540],
        'output_image_size': [512, 360],
    }

    from sim.forcemap_isaac import RandomTableScene, DatasetGenerator
    from omni.isaac.core import World

    world = World(stage_units_in_meters=1.0)
    scene = RandomTableScene(world, env_config)
    dataset = DatasetGenerator(scene)

    dataset.create(args.num_scenes, 3)


elif args.method == 'smoothing':
    # basket
    # env_config = {
    #     'data_dir': os.environ["HOME"] + "/Dataset/forcemap/basket240511",
    #     'object_set': "ycb_conveni_v1_small",
    #     'forcemap': 'seria_basket',
    #     'forcemap_bandwidth': 0.03,
    #     'scene_id': 81,
    # }

    # TRO
    # env_config = {
    #     'data_dir': os.environ["HOME"] + "/Dataset/forcemap/tabletop240304",
    #     'object_set': "ycb_conveni_v1",
    #     'forcemap': 'small_table',
    #     'forcemap_bandwidth': 0.03,
    #     'use_precomputed_sdfs': True,
    # }
    
    # AIREC
    env_config = {
        'data_dir': os.environ["HOME"] + "/Dataset/forcemap/tabletop_airec241008",
        'object_set': "ycb_conveni_v1",
        'forcemap': 'small_table',
        'forcemap_bandwidth': 0.03,
        'use_precomputed_sdfs': True,
    }

    from force_estimation import force_distribution_viewer
    from force_estimation.label_engineering import FmapSmoother
    import time
    from concurrent import futures

    viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
    # fmap_smoother = FmapSmoother(env_config, viewer)
    
    fmap_smoother = FmapSmoother(env_config)

    def compute_force_distribution_for_all(scene_numbers=range(0, 1000)):
        start_tm = time.time()
        with futures.ProcessPoolExecutor() as executor:
            executor.map(fmap_smoother.compute_force_distribution, scene_numbers)
        print(f"compute force distribution took: {time.time() - start_tm} [sec]")

    compute_force_distribution_for_all()

else:
    print('--method option is not specified')
