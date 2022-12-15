# -*- coding: utf-8 -*-

import force_estimation_v2 as fe


viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()

model_file = 'ae_cp.basket-filling2.model_resnet.20221202165608'

model = fe.model_rgb_to_fmap_res50()
test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
tester = Tester(model, test_data, model_file)
