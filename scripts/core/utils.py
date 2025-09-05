# -*- coding: utf-8 -*-

import os
import time
import json
import datetime

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
# import termcolor
import yaml
from PIL import Image
from scipy.linalg import norm


###
###
###


def swap(x):
    return x[1], x[0]


###
# Visualization tools
###


def draw_rect(image, roi):
    """
    To draw i'th image with ROI rectangle

    draw_rect(x[0][i][0], rois[i])
    plt.show()
    """
    fig, ax = plt.subplots()
    ax.imshow(image)
    height = image.shape[0]
    width = image.shape[1]
    x = width * roi[1]
    w = width * (roi[3] - roi[1])
    y = height * roi[0]
    h = height * (roi[2] - roi[0])
    rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", fill=False)  # x,y,w,h [pixels]
    ax.add_patch(rect)


# def draw_bounding_boxes(images, bboxes, color=[[1, 0, 0]]):
#     if type(images) != np.ndarray:
#         images = np.array(images)
#     if type(bboxes) != np.ndarray:
#         bboxes = np.array(bboxes)
#         if bboxes.ndim == 2:
#             bboxes = np.expand_dims(bboxes, 1)
#     return tf.image.draw_bounding_boxes(images, bboxes, color)


def create_anim_gif_from_images(images, out_filename, rois=[], predicted_images=[]):
    imgs = []
    if type(images[0]) == str:
        for img_file in images:
            img = Image.open(img_file)
            imgs.append(img)
    elif type(images[0]) == np.ndarray:
        if len(rois) > 0:
            imgs = draw_bounding_boxes(images, rois)
            if len(predicted_images) > 0:
                imgs = [np.concatenate([i.numpy(), pi], axis=0) for (i, pi) in zip(imgs, predicted_images)]
                imgs = [Image.fromarray((255 * i).astype(np.uint8)) for i in imgs]
        else:
            if len(predicted_images) > 0:
                imgs = [np.concatenate([i, pi], axis=0) for (i, pi) in zip(images, predicted_images)]
        imgs = [Image.fromarray((255 * i).astype(np.uint8)) for i in images]
    else:
        imgs = images
    imgs[0].save(out_filename, save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)


def create_anim_gif_for_group(images, rois, group_num, dir="./images"):
    n_frames = rois.shape[0]
    image_files = []
    for i in range(n_frames):
        draw_rect(images[i], rois[i])

        if not os.path.exists(dir):
            os.mkdir(dir)

        path = os.path.join(*[dir, str("{:05}.png".format(i))])
        plt.savefig(path)
        plt.close()
        image_files.append(path)
        create_anim_gif_from_images(image_files, "group{:05}.gif".format(group_num))


def visualize_ds(images, rois=[], max_samples=20, colorize_gray_image=True):
    if images.shape[-1] == 1 and (not colorize_gray_image):  # gray scale
        images = np.repeat(images, 3, axis=-1)
    else:
        images = np.squeeze(images)

    samples = min(len(images), max_samples)

    fig = plt.figure(figsize=(10, samples))
    fig.subplots_adjust(hspace=0.1)

    for p in range(samples):
        ax = fig.add_subplot(samples // 4, 4, p + 1)
        ax.axis("off")
        ax.imshow(images[p])
        if len(rois) > 0:
            roi = rois[p][0]
            height = images[p].shape[0]
            width = images[p].shape[1]
            x = width * roi[1]
            w = width * (roi[3] - roi[1])
            y = height * roi[0]
            h = height * (roi[2] - roi[0])
            rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="red", fill=False)  # x,y,w,h [pixels]
            ax.add_patch(rect)


def coordinate(axes, range_x, range_y, grid=True, xyline=True, xlabel="x", ylabel="y"):
    axes.set_xlabel(xlabel, fontsize=16)
    axes.set_ylabel(ylabel, fontsize=16)
    axes.set_xlim(range_x[0], range_x[1])
    axes.set_ylim(range_y[0], range_y[1])
    if grid is True:
        axes.grid()
    if xyline is True:
        axes.axhline(0, color="gray")
        axes.axvline(0, color="gray")


def draw_vector(axes, loc, vector, color="red", scale=1, width=1):
    axes.quiver(
        loc[0], loc[1], vector[0], vector[1], color=color, angles="xy", scale_units="xy", scale=scale, width=width
    )


def draw_trajectory(ax, traj, color="red", scale=0.4, width=0.002):
    for p1, p2 in traj:
        p1 = np.array(p1[:2])
        p2 = np.array(p2[:2])
        draw_vector(ax, p1, p2 - p1, color=color, scale=scale, width=width)


def draw_predictions_and_labels(ax, cs, ps, ls, colors=["red", "blue"], scale=0.4, width=0.002):
    trajs = [zip(cs[::3], ls[::3]), zip(cs[::3], ps[::3])]
    for traj, color in zip(trajs, colors):
        draw_trajectory(ax, traj, color=color, scale=scale, width=width)


###
# Data loader
###


def load_torobo_unity_joint_seq(joint_seq_file, start_step=0, step_length=None):
    joint_seq = []
    joint_time = []
    with open(joint_seq_file) as f:
        contents = f.readlines()
        if step_length is None:
            step_length = len(contents)
        for line_idx in range(start_step, start_step + step_length):
            line = contents[line_idx]
            line = line.rstrip("\n")
            line = line.rstrip(", )")
            line = line.split(":")
            time = line[2].rstrip("\tPOSITION")
            time_arr = np.fromstring(time, dtype=np.float64, sep=" ")
            data = line[3].rstrip(" )")
            data = data.lstrip(" (")
            data_arr = np.fromstring(data, dtype=np.float64, sep=", ")
            joint_time.append(time_arr)
            joint_seq.append(data_arr)
    np_joint_time = np.array(joint_time)
    np_joint_seq = np.array(joint_seq)
    np_joint_seq_time = np.concatenate([np_joint_time, np_joint_seq], 1)

    return np_joint_seq_time


class Dataset:
    def __init__(self, name, mode, **kargs):
        """
        Args:
            mode: 'train' or 'test'
        """
        self._name = name

        try:
            config_file = kargs["config_file"]
        except KeyError:
            config_file = "../../specification/training/dataset.yaml"
        with open(config_file, mode="r") as f:
            tree = yaml.safe_load(f)
            self._config = tree[name]

        try:
            self.dataset_path = kargs["dataset_path"]
        except KeyError:
            self.dataset_path = eval(tree["dataset-path"])

        self._directory = self._config["directory"]

        assert mode == "train" or mode == "test"
        if mode == "train":
            self._groups = eval(self._config["train_groups"])
        if mode == "test":
            self._groups = eval(self._config["validation_groups"])

        self.joint_range_data = eval(self._config["joint_range_data"])
        self.compute_joint_position_range()

    def compute_joint_position_range(self):
        self.load(groups=self.joint_range_data, load_image=False, normalize=False)
        jmaxs = np.max([np.max(group, axis=0) for group in self.data], axis=0)
        jmins = np.min([np.min(group, axis=0) for group in self.data], axis=0)
        self.joint_max_positions = np.where(np.isclose(jmaxs, jmins), 1, jmaxs)
        self.joint_min_positions = np.where(np.isclose(jmaxs, jmins), 0, jmins)

    def joint_position_range(self):
        return self.joint_min_positions, self.joint_max_positions

    def normalize_joint_position(self, q):
        return (q - self.joint_min_positions) / (self.joint_max_positions - self.joint_min_positions)

    def unnormalize_joint_position(self, q):
        return q * (self.joint_max_positions - self.joint_min_positions) + self.joint_min_positions

    def load_group(self, group, load_image=True, image_size=(90, 160), sampling_interval=1, normalize=True):
        path = os.path.join(self.dataset_path, "%s/%d" % (self._directory, group))
        # load joint and frame indices
        joint_file = os.path.join(path, "joint_position.txt")
        joint_seq = np.loadtxt(joint_file)

        if normalize:
            jmin = self.joint_min_positions
            jmax = self.joint_max_positions
            joint_seq = (joint_seq - jmin) / (jmax - jmin)

        # load images
        n_frames = joint_seq.shape[0]
        if load_image:
            frames = []
            for i in range(0, n_frames, sampling_interval):
                img = plt.imread(os.path.join(path, "image_frame%05d.jpg" % i))

                # img = img[130:250, 120:360] # crop the center
                img = cv2.resize(img, (image_size[1], image_size[0]))

                if normalize:
                    img = img / 255.0
                frames.append(img)

            return joint_seq, frames
        else:
            return joint_seq

    def load(
        self,
        image_size=(90, 160),
        groups=None,
        load_image=True,
        sampling_interval=1,
        visualize=False,
        start_step=0,
        step_length=None,
        normalize=True,
    ):
        start = time.time()
        data = []

        if groups is None:
            groups = self._groups

        n_groups = len(groups)
        for i, group in enumerate(groups):
            print("\rloading: {}/{}".format(i, n_groups), end="")
            data.append(
                self.load_group(
                    group,
                    load_image=load_image,
                    image_size=image_size,
                    sampling_interval=sampling_interval,
                    normalize=normalize,
                )
            )

        end = time.time()
        print("\ntotal time spent for loading data: {} [min]".format((end - start) / 60))
        self.data = data

    def get(self):
        return self.data

    @property
    def name(self):
        return self._name

    def extend_initial_and_final_states(self, size):
        def extend_group(gd):
            joint_seq, images = gd
            joint_seq2 = np.empty((joint_seq.shape[0] + size * 2,) + joint_seq.shape[1:])
            joint_seq2[:size] = joint_seq[0]
            joint_seq2[size : size + joint_seq.shape[0]] = joint_seq
            joint_seq2[size + joint_seq.shape[0] :] = joint_seq[-1]
            images2 = [images[0]] * size + images + [images[-1]] * size
            return joint_seq2, images2

        data2 = []
        for gd in self.data:
            data2.append(extend_group(gd))
        self.data = data2

    def smoothen(self, filter_size=5):
        """
        filter_size must be odd number
        """
        w = np.ones(filter_size) / filter_size

        def smoothen_jv(jv):
            seqlen, dim = jv.shape
            smoothed_jv = np.empty((seqlen - filter_size + 1, dim))
            for i in range(dim):
                jvp = np.convolve(jv[:, i], w, "valid")
                smoothed_jv[:, i] = jvp
            return smoothed_jv

        def smoothen_group(gd):
            joint_seq, images = gd
            smoothed_joint_seq = smoothen_jv(joint_seq)
            c = int((filter_size - 1) / 2)
            return smoothed_joint_seq, images[c:-c]

        data2 = []
        for gd in self.data:
            data2.append(smoothen_group(gd))
        self.data = data2

    def preprocess(self, extend_size):
        self.extend_initial_and_final_states(extend_size)
        self.smoothen()


##
# Analysis
##


def print_distances(joint_vec_seq):
    for i in range(len(joint_vec_seq) - 1):
        print(norm(joint_vec_seq[i] - joint_vec_seq[i + 1]))


class StateManager:
    def __init__(self, time_window_size, batch_size=1):
        self.history = []
        self.frames = []
        self.time_window_size = time_window_size
        self.batch_size = batch_size

    def initializeHistory(self, img, js):
        a = np.repeat(img[None, :], self.time_window_size, axis=0)
        a = np.repeat(a[None, :], self.batch_size, axis=0)
        b = np.repeat(js[None, :], self.time_window_size, axis=0)
        b = np.repeat(b[None, :], self.batch_size, axis=0)
        self.history = a, b
        self.frames = []

    def setHistory(self, images, joint_states):
        b = np.empty((self.batch_size,) + joint_states.shape)
        b[0] = joint_states
        a = np.empty((self.batch_size,) + images.shape)
        a[0] = images
        self.history = a, b

    def rollHistory(self, img, js):
        a = np.roll(self.history[0], -1, axis=1)
        a[:, -1] = img
        b = np.roll(self.history[1], -1, axis=1)
        b[:, -1] = js
        self.history = a, b

    def addState(self, img, js):
        self.rollHistory(img, js)
        self.frames.append((img, js))

    def getHistory(self):
        return self.history

    def getFrames(self):
        return self.frames

    def getFrameImages(self):
        return list(zip(*self.getFrames()))[0]

    def getFrameJointAngles(self):
        return list(zip(*self.getFrames()))[1]


##
# Print Message
##


def error(msg):
    print(termcolor.colored("[ERROR]: {}".format(msg), "red"))


def warn(msg):
    print(termcolor.colored("[WARN]: {}".format(msg), "yellow"))


def message(msg, tag=""):
    print(termcolor.colored("{}{}".format(tag, msg), "green"))


##
# EIPL style print functions
##

OK = '\033[92m'
WARN = '\033[93m'
NG = '\033[91m'
END_CODE = '\033[0m'

def print_info(msg):
    print( OK + "[INFO] " + END_CODE + msg )

def print_warn(msg):
    print( WARN + "[WARNING] " + END_CODE +  msg )

def print_error(msg):
    print( NG + "[ERROR] " + END_CODE + msg )

##
# EIPL utils
##
def check_path(path, mkdir=False):
    """
    checks given path is existing or not
    """
    if path[-1] == '/':
        path = path[:-1]

    if not os.path.exists(path):
        if mkdir:
            os.mkdir(path)
        else:
            raise ValueError("%s does not exist" % path)
    return path
                                                    

def set_logdir(log_dir, tag):
    return check_path(os.path.join(log_dir,tag), mkdir=True)


def normalization(data, indataRange, outdataRange):
    """
    Function to normalize a numpy array within a specified range
    Args:
        data (np.array): Data array
        indataRange (float list):  List of maximum and minimum values of original data, e.g. indataRange=[0.0, 255.0].
        outdataRange (float list): List of maximum and minimum values of output data, e.g. indataRange=[0.0, 1.0].
    Return:
        data (np.array): Normalized data array
    """
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0] )
    data = data * ( outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data          


def tensor2numpy(x):
    """
    Convert tensor to numpy array.
    """
    if x.device.type == 'cpu':
        return x.detach().numpy()
    else:
        return x.cpu().detach().numpy()
    
##
# EIPL path utils
##
# def check_filename(filename):
#     if os.path.exists(filename):
#         raise ValueError("{} exists.".format(filename))
#     return filename

# def check_path(path, mkdir=False):
#     """
#     Checks that path is collect
#     """
#     if path[-1] == '/':
#         path = path[:-1]

#     if not os.path.exists(path):
#         if mkdir:
#             os.makedirs(path, exist_ok=True)             
#         else:
#             raise ValueError('%s does not exist' % path)
    
#     return path


##
# EIPL arg utils
##

def print_args(args):
    """ Print arguments """
    if not isinstance(args, dict):
        args = vars(args)

    keys = args.keys() 
    keys = sorted(keys)

    print("================================")
    for key in keys:
        print("{} : {}".format(key, args[key]))
    print("================================")

    
def save_args(args, filename):
    """ Dump arguments as json file """
    with open(filename, 'w') as f:
        json.dump(vars(args), f, indent=4, sort_keys=True)

        
def restore_args(filename):
    """ Load argument file from file """
    with open(filename, 'r') as f:
        args = json.load(f)
    return args


def get_config(args, tag, default=None):
    """ Get value from argument """
    if not isinstance(args, dict):
        raise ValueError("args should be dict.")
    
    if tag in args:
        if args[tag] is None:
            print_info("set {} <-- {} (default)".format(tag, default))
            return default
        else:
            print_info("set {} <--- {}".format(tag, args[tag]))
            return args[tag]
    else:
        if default is None:
            raise ValueError("you need to specify config {}".format(tag))
        
        print_info("set {} <-- {} (default)".format(tag, default))
        return default

    
def check_args(args):
    """ Check arguments """

    if args.tag is None:
        tag = datetime.datetime.today().strftime("%Y%m%d_%H%M_%S")
        args.tag = tag
        print_info('Set tag = %s' % tag)
        
    # make log directory
    check_path(os.path.join(args.log_dir, args.tag), mkdir=True)

    # saves arguments into json file
    save_args(args, os.path.join(args.log_dir, args.tag, 'args.json'))
    
    print_args(args)
    return args