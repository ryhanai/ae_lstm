# -*- coding: utf-8 -*-

import os
import numpy as np
from force_estimation.force_estimation_data_loader import ForceEstimationDataLoader
from sim import run_sim_basket_filling
from functools import reduce
from operator import and_


dataset = 'basket-filling2'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62


dl = ForceEstimationDataLoader(os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset),
                               os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset+'-real'),
                               image_height=image_height,
                               image_width=image_width,
                               start_seq=1,
                               n_seqs=1800,  # n_seqs=1500,
                               start_frame=3, n_frames=3,
                               real_start_frame=1, real_n_frames=294
                               )


test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
bs = test_data[2]


def parse_bin_state(bs, i):
    return [int(str.split(o[0], '_')[0]) for o in bs[i]], i


def create_dataset(n_scenes=10,
                   objects = ['008_pudding_box',
                              '009_gelatin_box',
                              '010_potted_meat_can',
                              '011_banana',
                              '013_apple',
                              '026_sponge']):
    sw = run_sim_basket_filling.SceneWriter()
    for n in range(n_scenes):
        sw.createNewGroup()
        objects = np.random.permutation(objects)
        for object in objects:
            run_sim_basket_filling.place_object(object)
        sw.save_scene()
        run_sim_basket_filling.clear_scene()


def in_basket(pos):
    xrange = [-0.1, 0.1]
    yrange = [-.125, 0.125]
    zrange = [0.68, 0.73+0.15]
    return (xrange[0] < pos[0] < xrange[1]
            and yrange[0] < pos[1] < yrange[1]
            and zrange[0] < pos[2] < zrange[1])


def eval_force(k=1):
    """
    max force position
    """
    n_samples = 675
    imgs, fs, bs = test_data
    filtered_samples = []
    for i in range(n_samples):
        b = bs[i]
        if len(b) == 6 and reduce(and_, map(lambda x: in_basket(x[1][0]), b)):
            filtered_samples.append((i, np.max(fs[i]), np.average(fs[i]), imgs[i], fs[i], bs[i]))
    return sorted(filtered_samples, key=lambda x: x[k])

