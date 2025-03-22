from flask.cli import shell_command
import numpy as np
from fmdev.TabletopForceMapData import TabletopRandomSceneDataset

def mse_loss(d1, d2):
    return ((d1 - d2)**2).mean()

def distance_mask(sdf, shell_thickness):
    df = np.abs(sdf)
    return np.where(df <= shell_thickness, 1., 0.)

def pc_loss1(d1, d2, sdf, shell_thickness=0.01):
    m = distance_mask(sdf, shell_thickness=shell_thickness)
    l = (m * (d1 - d2)**2)[m.nonzero()].mean()
    return l

def normalized_mse_loss(d_pred, d_gt):
    return mse_loss(d_pred, d_gt) / d_gt.mean()

def normalized_pc_loss1(d_pred, d_gt, sdf, shell_thickness=0.01):
    msk = distance_mask(sdf, shell_thickness=shell_thickness)
    l = (msk * (d_pred - d_gt)**2)[msk.nonzero()].mean()
    l /= (msk * d_gt)[msk.nonzero()].mean()
    return l


from fmdev.test_torch import *

dataset_path = '~/Dataset/forcemap'
task_name = 'tabletop240304'

weight_files = {
    'GAFS_f0.030_g0.010':'log/20250322_1023_08/00199.pth',
    'GAFS_f0.060_g0.010': 'log/20250322_1140_56/00199.pth',
    'IFS_f0.015': 'log/20250322_1043_24/00199.pth',
    'IFS_f0.005': 'log/20250322_1016_28/00199.pth'
}


class Experiment:
    def __init__(self):
        self._sdf_ds = TabletopRandomSceneDataset('test', task_name=task_name, method='sdf')
        self.clear_score()

    def clear_score(self):
        self._scores = {}
        self._nscores = {}

    def set_ckpt(self, weight_file):
        self._tester = Tester(dataset_path,
                              task_name,
                              [weight_file],
                              data_split='test')

    def eval_model(self, weight_file):
        self.set_ckpt(weight_file)
        score = []
        nscore = []
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

            nloss1 = normalized_mse_loss(d_predicted, d_label)
            nloss2 = normalized_pc_loss1(d_predicted, d_label, sdf, shell_thickness=0.010)
            nloss3 = normalized_pc_loss1(d_predicted, d_label, sdf, shell_thickness=0.005)            
            nscore.append((nloss1, nloss2, nloss3))

            print(f'{scene_idx}: MSE Loss={loss1}, PC Loss(0.01)={loss2}, PC Loss(0.005)={loss3}')
            print(f'{scene_idx}: nMSE Loss={nloss1}, nPC Loss(0.01)={nloss2}, nPC Loss(0.005)={nloss3}')            

        return score, nscore

    def eval_all_models(self, weight_files):
        for tag, weight_file in weight_files.items():
            score, nscore = self.eval_model(weight_file)
            self._scores[tag] = score
            self._nscores[tag] = nscore

    def format_tag(self, tag):
        abc = tag.split('_')
        if len(abc) == 3:
            a, b, c = abc
            return f'{a}$(\sigma_f={b[1:]},\sigma_g={c[1:]})$' 
        elif len(abc) == 2:
            a, b = abc
            return f'{a}$(\sigma_f={b[1:]})$'
        else:
            return tag

    def print_score(self, scores, caption='prediction losses'):
        print('\\begin{table*}[ht]')
        print('\\rowcolors{2}{white}{Gainsboro}')
        print('\\centering')
        print('\\begin{tabular}{ l|lll }')
        print('\\toprule')
        print('\\textbf{Smoothing Method} & \\textbf{MSE} & \\textbf{MSE, 1.0(cm)} & \\textbf{MSE, 0.5(cm)} \\\\ \midrule')

        for tag, vals in scores.items():
            vals = np.array(vals)
            means = np.average(vals, axis=0)
            stds = np.std(vals, axis=0)
            print(f'{self.format_tag(tag)} & ${means[0]:.5f} \pm {stds[0]:.5f}$ & ${means[1]:.5f} \pm {stds[1]:.5f}$ & ${means[2]:.5f} \pm {stds[2]:.5f}$ \\\\')

        print('\\bottomrule')
        print('\\end{tabular}')
        print(f'\\caption{{\\textbf{{{caption}}}}}')
        print('\\label{tab:prediction_losses}')
        print('\\end{table*}')

    def print_scores(self):
        self.print_score(self._scores, caption='prediction losses')
        self.print_score(self._nscores, caption='prediction losses (normalized)')


if __name__ == '__main__':
    ex = Experiment()
