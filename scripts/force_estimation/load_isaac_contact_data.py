# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# import scipy


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


from force_estimation import forcemap
from force_estimation import force_distribution_viewer


fmap = forcemap.GridForceMap('seria_basket')
viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()
# a = load_contact_data()
# d = fmap.getDensity(a[0], a[1])

def load_data(frameNo=0, scale=1e+4):
    d = pd.read_pickle('../sim/data/force_zip{:05d}.pkl'.format(frameNo))
    bin_state = pd.read_pickle('../sim/data/bin_state{:05d}.pkl'.format(frameNo))
    fmap.set_values(d*scale)
    viewer.publish_bin_state(bin_state, fmap)
