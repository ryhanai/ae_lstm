# from flask.cli import shell_command
from tkinter import W

import numpy as np
from regex import F
from scipy.spatial.transform import Rotation as R
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


from dataset.object_loader import ObjectInfo
object_info = ObjectInfo('ycb_conveni_v1', split='all')


from fmdev.test_torch import *

dataset_path = '~/Dataset/forcemap'

# task_name = 'tabletop240304'
# weight_files = {
#     'GAFS_f0.030_g0.010':'log/20250322_1023_08/00199.pth',
#     'GAFS_f0.060_g0.010': 'log/20250322_1140_56/00199.pth',
#     'IFS_f0.015': 'log/20250322_1043_24/00199.pth',
#     'IFS_f0.005': 'log/20250322_1016_28/00199.pth'
# }

task_name = 'tabletop250902'
weight_files = {
    'GAFS_f0.030_g0.010':'log/20250903_1152_46/00199.pth',
    'GAFS_f0.060_g0.010': 'log/20250903_1215_49/00199.pth',
    'IFS_f0.015': 'log/20250903_1234_37/00199.pth',
    'IFS_f0.005': 'log/20250903_1304_01/00199.pth'
}


class Experiment:
    def __init__(self):
        self._sdf_ds = TabletopRandomSceneDataset('test', task_name=task_name, method='sdf')
        self._gt_ds = TabletopRandomSceneDataset('test', task_name=task_name, sigma_f=0.005, method='isotropic')
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
            d_predicted = self._tester.predict(scene_idx, show_result=False)[0]
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
            return f'{a}$(\\sigma_f={b[1:]},\\sigma_g={c[1:]})$' 
        elif len(abc) == 2:
            a, b = abc
            return f'{a}$(\\sigma_f={b[1:]})$'
        else:
            return tag

    def print_score(self, scores, caption='prediction losses'):
        print('\\begin{table*}[ht]')
        print('\\rowcolors{2}{white}{Gainsboro}')
        print('\\centering')
        print('\\begin{tabular}{ l|lll }')
        print('\\toprule')
        print('\\textbf{Smoothing Method} & \\textbf{MSE} & \\textbf{MSE, 1.0(cm)} & \\textbf{MSE, 0.5(cm)} \\\\ \\midrule')

        for tag, vals in scores.items():
            vals = np.array(vals)
            means = np.average(vals, axis=0)
            stds = np.std(vals, axis=0)
            print(f'{self.format_tag(tag)} & ${means[0]:.5f} \\pm {stds[0]:.5f}$ & ${means[1]:.5f} \\pm {stds[1]:.5f}$ & ${means[2]:.5f} \\pm {stds[2]:.5f}$ \\\\')

        print('\\bottomrule')
        print('\\end{tabular}')
        print(f'\\caption{{\\textbf{{{caption}}}}}')
        print('\\label{tab:prediction_losses}')
        print('\\end{table*}')

    def print_scores(self):
        self.print_score(self._scores, caption='prediction losses')
        self.print_score(self._nscores, caption='prediction losses (normalized)')

    def get_dataset(self, model_index):
        return self._tester._model_dataset_pairs[model_index][1]


def unnormalize_force(fmap, force_bounds):
    fmap = np.exp(fmap)
    fmap = np.clip(fmap, force_bounds[0], force_bounds[1])
    return fmap


def pose_to_matrix(position, quaternion):
    R_mat = R.from_quat(quaternion).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R_mat
    T[:3, 3] = position
    return T


def get_CoM(object_name, obj_pose):
    com_local = object_info.CoM(object_name)
    com_world = obj_pose[:3, :3] @ com_local + obj_pose[:3, 3]
    return com_world

def get_center(object_name, obj_pose):
    center_local = object_info.center(object_name)
    center_world = obj_pose[:3, :3] @ center_local + obj_pose[:3, 3]
    return center_world


def unnormalize_force(fvals, ds):
    return np.exp(normalization(fvals, ds.minmax, np.log(ds._force_bounds)))


def fibonacci_sphere(n):
    i = np.arange(n)
    phi = np.arccos(1 - 2*(i + 0.5)/n)
    theta = np.pi * (1 + 5**0.5) * (i + 0.5)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    return np.stack([x, y, z], axis=1)


def eval_resistance(d, force_vecs):
    w = np.dot(force_vecs, d)
    return -np.sum(np.where(w < 0, w, 0))


def collect_all_contact_forces_for_object(contact_data, target_object):
    """
    指定した物体に関係する接触力を集める。
    返す direction は「target_object に働く向き」に揃える。

    ルール:
    - pair = (A, B) は「A から B に力が働く」
    - target_object == B のとき:
        そのまま採用
    - target_object == A のとき:
        反作用なので direction を反転して採用

    Returns:
        list of (position, magnitude, direction)
    """
    positions, magnitudes, pairs, directions = contact_data

    force_list = []
    for pos, mag, pair, direction in zip(positions, magnitudes, pairs, directions):
        src_obj, dst_obj = pair
        pos = np.asarray(pos)
        direction = np.asarray(direction, dtype=np.float32)

        if dst_obj == target_object:
            force_list.append((pos, float(mag), direction))
        elif src_obj == target_object:
            force_list.append((pos, float(mag), -direction))

    return force_list


ex = Experiment()


def evaluate(model_idx=0,
             scene_idx=0,
             object_name='007_tuna_fish_can',
             shell_thickness=0.01,
             contact_threshold=1e-4,
             recall_percentile=90,
             resist_percentile=10,
             ):

    ex.set_ckpt(weight_file=list(weight_files.values())[model_idx])
    dt = 1/60.

    ds = ex._tester._model_dataset_pairs[0][1]
    bs = dict(ds.load_bin_state(scene_idx))
    obj_pose = bs[object_name]

    y_pred = ex._tester.predict(scene_idx, show_result=False)[0]
    sdf = ex._sdf_ds.load_sdf(scene_idx, object_name=object_name, transpose=False)  # array of shape (80, 80, 40)

    fv = np.zeros((80, 80, 40))
    # fv[:, :, :30] = y_pred
    fv[:, :, :30] = sdf  # use the geometry-based gradient as force directions
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])    

    fmap = ex._tester._fmap

    shell_mask = distance_mask(sdf, shell_thickness=shell_thickness)  # binary mask of shape (80, 80, 40)

    # convert to 1D array of force values and mask
    fmap.set_values(y_pred)
    force_values = fmap.get_values()
    force_values = unnormalize_force(force_values, ds) / dt  # impulse -> Force [N]

    force_vecs = g_vecs / (np.linalg.norm(g_vecs, axis=1, keepdims=True) + 1e-10) * force_values[:, None]

    # extract forces only on the shell
    fmap.set_values(shell_mask)
    shell_mask = fmap.get_values()
    masked_force_vecs = force_vecs[shell_mask.astype(bool)]
    masked_xy_coords = ex._tester._fmap.get_positions()[shell_mask.astype(bool)]

    # extract inward forces
    object_center = get_center(object_name, pose_to_matrix(*obj_pose))
    inward_mask = np.where(np.sum(masked_force_vecs * (masked_xy_coords - object_center), axis=1) < 0.0, 1, 0)
    approximated_force_vecs = masked_force_vecs[inward_mask.astype(bool)]
    approximated_force_poss = masked_xy_coords[inward_mask.astype(bool)]

    # Gravity Support
    total_force_on_object = approximated_force_vecs.sum(axis=0)
    mg = object_info.mass(object_name) * np.array([0, 0, -9.81])
    gravity_support = np.linalg.norm(total_force_on_object + mg) / np.linalg.norm(mg)
    print(f"Gravity support: {gravity_support}[{total_force_on_object}/{mg}]")

    # Torque Balance
    com = get_CoM(object_name, pose_to_matrix(*obj_pose))
    torques = np.cross(approximated_force_poss - com, approximated_force_vecs)
    tau_res = np.linalg.norm(np.sum(torques, axis=0))
    tau_total = np.sum(np.linalg.norm(torques, axis=1))
    relative_torque_balance = tau_res / (tau_total + 1e-8)
    print(f"Torque balance: {relative_torque_balance}")

    # Agreement of contact regions
    gt_point_forces = ex._gt_ds.load_fmap(scene_idx, transpose=False)
    force_values_gt = np.zeros((80, 80, 40))
    force_values_gt[:, :, :30] = gt_point_forces
    fmap.set_values(force_values_gt)
    force_values_gt = fmap.get_values()
    force_values_gt = unnormalize_force(force_values_gt, ds) / dt  # impulse -> Force [N]

    contact_gt = np.where(force_values_gt >= contact_threshold, 1., 0.)
    contact_pred = np.where(force_values >= contact_threshold, 1., 0.)

    contact_I = (contact_pred.astype(bool) & contact_gt.astype(bool)).sum() 
    contact_U =(contact_pred.astype(bool) | contact_gt.astype(bool)).sum()
    contact_recall = contact_I / contact_gt.astype(bool).sum()
    contact_IoU = contact_I / contact_U
    # print(f"Contact_IoU: {contact_IoU}[{contact_I}/{contact_U}]")
    # print(f"Contact Recall: {contact_recall}")

    # agreement of low resistance directions
    directions = fibonacci_sphere(100)
    resistances = []
    for d in directions:
        resistances.append(eval_resistance(d, approximated_force_vecs))

    contact_data = ds.load_raw_contact_data(scene_idx)
    raw_forces = collect_all_contact_forces_for_object(contact_data, object_name)
    raw_fvecs = []
    raw_fposs = []
    for p, v, d in raw_forces:
        raw_fvecs.append(v * d)
        raw_fposs.append(p)
    raw_fvecs = np.array(raw_fvecs)
    raw_fposs = np.array(raw_fposs)
    raw_fvecs /= dt
    raw_fvecs = -raw_fvecs  # for some reason direction was opposite

    resistances_gt = []
    for d in directions:
        resistances_gt.append(eval_resistance(d, raw_fvecs))
    
    low_resist_pred = np.where(resistances <= np.percentile(resistances, resist_percentile), 1., 0.)
    low_resist_gt = np.where(resistances_gt <= np.percentile(resistances_gt, resist_percentile), 1., 0.)    

    low_resist_I = (low_resist_pred.astype(bool) & low_resist_gt.astype(bool)).sum() 
    low_resist_U = (low_resist_pred.astype(bool) | low_resist_gt.astype(bool)).sum()
    low_resist_IoU = low_resist_I / low_resist_U
    print(f"LowResistanceDirection_IoU: {low_resist_IoU}[{low_resist_I}/{low_resist_U}]")
    print(f"LowResist[pred/gt]: {low_resist_pred.sum()}, {low_resist_gt.sum()}")

    return directions, resistances, resistances_gt

    # return approximated_force_poss, approximated_force_vecs
    # return contact_pred, contact_gt, approximated_force_poss, approximated_force_vecs

    
def evaluate_models(model_indices=range(4),
                    scene_idx=0,
                    object_name='007_tuna_fish_can',
                    shell_thickness=0.01,
                    contact_threshold=1e-4,
                    recall_percentile=90,
                    resist_percentile=10,
                 ):
    for model_idx in model_indices:
        evaluate(model_idx, scene_idx, object_name, shell_thickness, contact_threshold, recall_percentile, resist_percentile)
