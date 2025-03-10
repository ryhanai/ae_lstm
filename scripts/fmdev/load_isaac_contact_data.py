# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import glob
import os
import re
import time
from concurrent import futures


# rotate the basket by 90 degrees in Isaac
# object name must be output from Isaac
# impulse values are smaller in Isaac


# object_labels = [
#     '009_gelatin_box',
#     '005_tomato_soup_can',
#     '010_potted_meat_can',
#     '011_banana',
#     '013_apple',
#     '061_foam_brick'
# ]


# def load_contact_data(file='/home/ryo/Downloads/contacts.zip'):
#     bin_state = []
#     contacts = []
#     data = pd.read_pickle(file)
#     for i in range(len(data)):
#         c, pose = data[i]
#         contacts.extend(c)
#         object_label = object_labels[i]
#         bin_state.append((object_label, pose))

#     filtered_contacts = list(filter(lambda x: scipy.linalg.norm(x['impulse']) > 1e-8, contacts))
#     contact_positions = np.array([x['position'] for x in filtered_contacts])
#     impulse_value = np.array([scipy.linalg.norm(x['impulse']) for x in filtered_contacts])

#     return contact_positions, impulse_value, bin_state


from fmdev import forcemap
from fmdev import force_distribution_viewer


# data_dir = '../sim/data'
data_dir = f"{os.environ['HOME']}/Dataset/dataset2/basket-filling3"
scene = 'seria_basket'

# fmap = forcemap.GridForceMap(scene)
fmap = forcemap.GridForceMap(scene)
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
# a = load_contact_data()
# d = fmap.getDensity(a[0], a[1])

def load_data(frameNo=0, scale=1):
    d = pd.read_pickle(f'{data_dir}/force_zip{frameNo:05}.pkl'.format(frameNo))
    bin_state = pd.read_pickle(f'{data_dir}/bin_state{frameNo:05}.pkl'.format(frameNo))
    fmap.set_values(d*scale)
    viewer.publish_bin_state(bin_state, fmap)
    print(bin_state)
    return d, bin_state


def compute_force_distribution(contact_raw_data_file, log_scale=True, overwrite=False):
    local_fmap = forcemap.GridForceMap(scene)
    frameNo = int(re.search('\d+', os.path.basename(contact_raw_data_file)).group())
    out_file = os.path.join(data_dir, 'force_zip{:05d}.pkl'.format(frameNo))
    if (not overwrite) and os.path.exists(out_file):
        print(f'skip [{frameNo}]')
    else:
        print(f'process [{frameNo}]')
        contact_positions, impulse_values = pd.read_pickle(contact_raw_data_file)
        d = local_fmap.getDensity(contact_positions, impulse_values, return_3d=True)
        if log_scale:
            d = np.log(1 + d)
        pd.to_pickle(d, out_file)


def compute_force_distribution_for_all():
    start_tm = time.time()
    contact_raw_data_files = glob.glob(os.path.join(data_dir, 'contact_raw_data*.pkl'))
    with futures.ProcessPoolExecutor() as executor:
        executor.map(compute_force_distribution, contact_raw_data_files)
    print(f'compute force distribution took: {time.time() - start_tm} [sec]')