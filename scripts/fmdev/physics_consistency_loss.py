import numpy as np
from fmdev.TabletopForceMapData import TabletopRandomSceneDataset


def distance_mask(sdf, shell_thickness):
    df = np.abs(sdf)
    return np.where(df <= shell_thickness, 1., 0.)

def pc_loss1(d1, d2, sdf, shell_thickness=0.01):
    m = distance_mask(sdf, shell_thickness=shell_thickness)
    l = (m * (d1 - d2)**2)[m.nonzero()].mean()
    return l

def mse_loss(d1, d2):
    return ((d1 - d2)**2).mean()


from fmdev.test_torch import *

dataset_path = '~/Dataset/forcemap'
task_name = 'tabletop240304'

weight_files = {
    'GAFS_f0.030_g0.010':'log/20250321_0935_35/00199.pth',
    'GAFS_f0.060_g0.010': 'log/20250321_0141_06/00199.pth',
    'IFS_f0.015': 'log/20250321_1423_23/00199.pth',
    'IFS_f0.005': 'log/20250321_1243_40/00199.pth',
}


class Experiment:
    def __init__(self):
        self._sdf_ds = TabletopRandomSceneDataset('test', task_name=task_name, method='sdf')
        self.clear_score()

    def clear_score(self):
        self._scores = {}

    def set_ckpt(self, weight_file):
        self._tester = Tester(dataset_path,
                              task_name,
                              [weight_file],
                              data_split='test')

    def eval_model(self, weight_file):
        self.set_ckpt(weight_file)
        score = []
        model, ds = self._tester._model_dataset_pairs[0]

        for scene_idx in range(ds.__len__()):
        # for scene_idx in range(3):            
            d_predicted = self._tester.predict(scene_idx)[0]
            sdf = self._sdf_ds.load_fmap(scene_idx).transpose(1, 2, 0)
            d_label = ds.load_fmap(scene_idx).transpose(1, 2, 0)
            loss1 = mse_loss(d_label, d_predicted)
            loss2 = pc_loss1(d_label, d_predicted, sdf, shell_thickness=0.010)
            loss3 = pc_loss1(d_label, d_predicted, sdf, shell_thickness=0.005)
            score.append((loss1, loss2, loss3))
            print(f'{scene_idx}: MSE Loss={loss1}, PC Loss(0.005)={loss2}, PC Loss(0.002)={loss3}')

        return score

    def eval_all_models(self, weight_files):
        for tag, weight_file in weight_files.items():
            self._scores[tag] = self.eval_model(weight_file)

    def print_score(self):
        print('\\begin{table}[ht]')
        print('\\rowcolors{2}{white}{Gainsboro}')
        print('\\centering')
        print('\\begin{tabular}{ l|lll }')
        print('\\toprule')
        print('\\textbf{Smoothing} & \\textbf{MSE} & \\textbf{MSE, 1.0(cm)} & \\textbf{MSE, 0.5(cm)} \\\\ \midrule')

        for tag, vals in self._scores.items():
            vals = np.array(vals)
            means = np.average(vals, axis=0)
            stds = np.std(vals, axis=0)
            print(f'{tag} & {means[0]:.5f} \pm {stds[0]:.5f} & {means[1]:.5f} \pm {stds[1]:.5f} & {means[2]:.5f} \pm {stds[2]:.5f} \\\\')

        print('\\bottomrule')
        print('\\end{tabular}')
        print('\\caption{\\textbf{prediction losses}}')
        print('\\label{tab:prediction_losses}')
        print('\\end{table}')


ex = Experiment()
