# -*- coding: utf-8 -*-

import os, sys, glob, re, time, copy
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from generator import *
from PIL import Image

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

def create_anim_gif(images, out_filename):
    imgs = []
    if type(images[0]) == str:
        for img_file in img_files:
            img = Image.open(img_file)
            imgs.append(img)
    elif type(images[0]) == np.ndarray:
        imgs = [Image.fromarray((255*i).astype(np.uint8)) for i in images]
    else:
        imgs = images
    imgs[0].save(out_filename, save_all=True, append_images=imgs[1:], optimize=False, duration=100, loop=0)

def draw_rect(image, roi):
    '''
    To draw i'th image with ROI rectangle

    draw_rect(x[0][i][0], rois[i])
    plt.show()
    '''
    fig,ax = plt.subplots()
    ax.imshow(image)
    height = image.shape[0]
    width = image.shape[1]
    x = width * roi[1]
    w = width * (roi[3] - roi[1])
    y = height * roi[0]
    h = height * (roi[2] - roi[0])
    rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='red', fill=False) # x,y,w,h [pixels]
    ax.add_patch(rect)

def create_anim_gif_for_group(images, rois, group_num, dir='./images'):
    n = rois.shape[0]
    image_files = []
    for i in range(n):
        draw_rect(images[i], rois[i])

        if os.path.exists(dir):
            pass
        else:
            os.mkdir(dir)

        path = os.path.join(*[dir, str('{:05}.png'.format(i))])
        plt.savefig(path)
        plt.close()
        image_files.append(path)
        create_anim_gif(image_files, 'group{:05}.gif'.format(group_num))

def visualize_ds(images, rois=[], max_samples=20):
    samples = min(len(images), max_samples)

    fig = plt.figure(figsize=(10,samples))
    fig.subplots_adjust(hspace=0.1)

    for p in range(samples):
        ax = fig.add_subplot(samples//4, 4, p+1)
        ax.axis('off')
        ax.imshow(images[p])
        if len(rois) > 0:
            roi = rois[samples][0]
            height = images[p].shape[0]
            width = images[p].shape[1]
            x = width * roi[1]
            w = width * (roi[3] - roi[1])
            y = height * roi[0]
            h = height * (roi[2] - roi[0])
            rect = patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='red', fill=False) # x,y,w,h [pixels]
            ax.add_patch(rect)

dataset_path = '/home/ryo/Dataset/dataset2'
sys.path.append(dataset_path)

roi_shape = (40, 80, 3)
roi_size = roi_shape[:2]
input_image_shape = (90, 160, 3)
input_image_size = input_image_shape[:2]

def load_torobo_unity_joint_seq(joint_seq_file, start_step=0, step_length=None):
    joint_seq = []
    joint_time = []
    with open(joint_seq_file) as f:
        contents = f.readlines()
        if step_length == None:
            step_length = len(contents)
        for line_idx in range(start_step, start_step+step_length):
            line = contents[line_idx]
            line = line.rstrip('\n')
            line = line.rstrip(', )')
            line = line.split(':')
            time = line[2].rstrip('\tPOSITION')
            time_arr = np.fromstring(time, dtype=np.float64, sep=' ')
            data = line[3].rstrip(' )')
            data = data.lstrip(' (')
            data_arr = np.fromstring(data, dtype=np.float64, sep=', ')
            joint_time.append(time_arr)
            joint_seq.append(data_arr)
    np_joint_time = np.array(joint_time)
    np_joint_seq = np.array(joint_seq)
    np_joint_seq_time = np.concatenate([np_joint_time, np_joint_seq], 1)

    return np_joint_seq_time


def load_dataset(action='pushing',
                 groups=range(1,6),
                 load_image=True,
                 sampling_interval=1,
                 visualize=False,
                 start_step=0,
                 step_length=None,
                 normalize=True):

    def load_group(group, joint_range=None):
        path = os.path.join(dataset_path, '%s/%d'%(action, group))

        # load joint and frame indices
        joint_file = os.path.join(path, 'joint_position.txt')
        joint_seq = np.loadtxt(joint_file)

        if normalize:
            jmin, jmax = joint_range
            joint_seq = (joint_seq - jmin) / (jmax - jmin)

        # load images
        n_frames = joint_seq.shape[0]
        if load_image:
            frames = []
            for i in range(0, n_frames, sampling_interval):
                img = plt.imread(os.path.join(path, 'image_frame%05d.jpg'%i))
                # img = img[130:250, 120:360] # crop the center
                img = cv2.resize(img, (input_image_size[1], input_image_size[0]))

                if normalize:
                    img = img/255.
                frames.append(img)

            return joint_seq, frames
        else:
            return joint_seq

    start = time.time()
    data = []

    if normalize:
        jp_range = joint_position_range()

    n_groups = len(groups)
    for i,group in enumerate(groups):
        print('\rloading: {}/{}'.format(i, n_groups), end='')
        if normalize:
            data.append(load_group(group, jp_range))
        else:
            data.append(load_group(group))

    end = time.time()
    print('\ntotal time spent for loading data: {} [min]'.format((end-start)/60))

    return data

def joint_position_range():
    ds = load_dataset(groups=range(1,400), load_image=False, normalize=False)
    joint_max_positions = np.max([np.max(group, axis=0) for group in ds], axis=0)
    joint_min_positions = np.min([np.min(group, axis=0) for group in ds], axis=0)
    return joint_min_positions, joint_max_positions

def generate_dataset_for_AE_training(ds):
    images = []
    for group in ds:
        joint_seq, frames = group
        images.extend(frames)
    input_images = np.array(images) / 255.0
    labels = np.zeros((input_images.shape[0], 80, 160, 3))
    roi = np.array([0.48, 0.25, 0.92, 0.75]) # [y1, x1, y2, x2] in normalized coodinates
    bboxes = np.tile(roi, [input_images.shape[0], 1])
    return (input_images, bboxes), labels

# def fixed_ROI_policy(ds):
#     roi = np.array([0.48, 0.25, 0.92, 0.75])
#     n_data = [group[0].shape[0] for group in ds]
#     return np.tile(roi, [ds.shape[0], 1])


from scipy.linalg import norm

def print_distances(joint_vec_seq):
    for i in range(len(joint_vec_seq)-1):
        print(norm(joint_vec_seq[i] - joint_vec_seq[i+1]))


def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, roi_size)

def model_crop_and_resize():
    image_input = tf.keras.Input(shape=input_image_shape)
    roi_input = tf.keras.Input(shape=(4, ))
    cropped = tf.keras.layers.Lambda(crop_and_resize, output_shape=roi_size) ([image_input, roi_input])
    return tf.keras.Model([image_input, roi_input], cropped, name='test')

def test_crop_and_resize():
    model = model_crop_and_resize()
    imgs = np.array(load_dataset(groups=[2])[0][1]) / 255
    rois = np.tile([0.2, 0.2, 0.8, 0.8], [imgs.shape[0], 1])
    result = model([imgs, rois])
    plt.imshow(result[50])
    plt.show()


loss_tracker = keras.metrics.Mean(name="loss")
val_loss_tracker = keras.metrics.Mean(name="val_loss")

dense_dim = 512
latent_dim = 32
time_window_size = 10
dof = 7
batch_size = 32

class AutoencoderWithCrop(tf.keras.Model):
    """
    override loss computation
    """
    def reconst_roi(self):
        return np.tile(np.array([0.0, 0.0, 1.0, 1.0]), [batch_size, 1])

    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True) # Forward pass
            loss = self.compute_loss(y, y_pred, x)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        loss_tracker.update_state(loss)
        return {"loss": loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass

        roi = y_pred[2]
        print(' ROI: ', roi)

        val_loss = self.compute_loss(y, y_pred, x)

        val_loss_tracker.update_state(val_loss)
        return {"loss": val_loss_tracker.result()}

    def compute_loss(self, y, y_pred, x):
        x_image, x_joint = x
        y_image, y_joint = y
        roi = y_pred[2]
        #y_image_cropped = crop_and_resize((y_image, roi[:,-1]))
        y_image_cropped = crop_and_resize((y_image, self.reconst_roi()))

        image_loss = tf.reduce_mean(tf.square(y_image_cropped - y_pred[0]))
        joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))
        loss = image_loss + joint_loss
        #loss = image_loss + 10.*joint_loss
        #loss = keras.losses.mean_squared_error(y_image_cropped, y_pred[0])
        return loss

    @property
    def metrics(self):
        return [loss_tracker, val_loss_tracker]

def const_roi_fun():
    center = np.array([0.7, 0.5])
    v = np.array([0.22, 0.25])
    roi = np.concatenate((center-v, center+v)) # [y1, x1, y2, x2] in normalized coodinates
    return roi

# def roi_rect(args):
#     c, s = args
#     es = 0.15 * (1.0 + s)
#     #es = 0.075 * (1.0 + s)
#     img_center = tf.tile(tf.constant([[0.5, 0.5]], dtype=tf.float32), (batch_size,1))
#     a = tf.tile(tf.expand_dims(es, 1), (1,2))
#     lt = img_center + 0.4 * (c - img_center) - a
#     rb = img_center + 0.4 * (c - img_center) + a
#     roi = tf.concat([lt, rb], axis=1)
#     roi3 = tf.expand_dims(roi, 0)
#     return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

def roi_rect(args):
    c, s = args
    lt = tf.tile(tf.constant([[0.0, 0.0]], dtype=tf.float32), (batch_size,1))
    rb = tf.tile(tf.constant([[1.0, 1.0]], dtype=tf.float32), (batch_size,1))
    roi = tf.concat([lt, rb], axis=1)
    roi3 = tf.expand_dims(roi, 0)
    return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

def model_encoder():
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(0.2))(image_input)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    encoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_dim, activation='selu'))(x)

    encoder = keras.Model([image_input], encoder_output, name='encoder')
    encoder.summary()
    return encoder

def model_lstm():
    # stacked LSTM
    imgvec_input = tf.keras.Input(shape=(time_window_size, latent_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = latent_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(50)(x)
    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(x) # (x, y, s)

    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])

    lstm = keras.Model([imgvec_input, joint_input], roi, name='lstm')
    lstm.summary()
    return lstm

def model_roi_encoder():
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    roi_input = tf.keras.Input(shape=(time_window_size, 4))

    roi = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(crop_and_resize, output_shape=roi_shape)) ([image_input, roi_input])
    roi_extractor = tf.keras.Model([image_input, roi_input], roi, name='roi_extractor')
    roi_extractor.summary()

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.GaussianNoise(0.2))(roi)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(8, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(16, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, kernel_size=3, strides=1, padding='same', activation='selu'))(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.BatchNormalization())(x)
    x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPool2D(pool_size=2))(x)

    x = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(x)
    encoder_output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(latent_dim, activation='selu'))(x)

    encoder = keras.Model([image_input, roi_input], encoder_output, name='roi_encoder')
    encoder.summary()
    return encoder

def model_roi_lstm():
    # stacked LSTM
    imgvec_input = tf.keras.Input(shape=(time_window_size, latent_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = latent_dim + dof
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    x = tf.keras.layers.LSTM(50)(x)
    x = tf.keras.layers.Dense(state_dim)(x)
    lstm_image_output = tf.keras.layers.Lambda(lambda x:x[:,:latent_dim], output_shape=(latent_dim,))(x)
    lstm_joint_output = tf.keras.layers.Lambda(lambda x:x[:,latent_dim:], output_shape=(dof,))(x)

    lstm = keras.Model([imgvec_input, joint_input], [lstm_image_output, lstm_joint_output], name='roi_lstm')
    lstm.summary()
    return lstm

def model_roi_decoder():
    channels = 64
    ivec_input = tf.keras.Input(shape=(latent_dim))
    x = tf.keras.layers.Dense(5*10*channels, activation='selu')(ivec_input)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Reshape(target_shape=(5, 10, channels))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)

    x = tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.UpSampling2D()(x)
    decoder_output = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    roi_decoder = keras.Model(ivec_input, decoder_output, name='roi_decoder')
    roi_decoder.summary()
    return roi_decoder

def model_roi_ae_lstm():
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    #roi_input = tf.keras.Input(shape=(time_window_size, 4))

    encoded_img = model_encoder()(image_input)
    roi = model_lstm()([encoded_img, joint_input])

    encoded_roi = model_roi_encoder()([image_input, roi])
    predicted_ivec, predicted_jvec = model_roi_lstm()([encoded_roi, joint_input])
    decoded_roi = model_roi_decoder()(predicted_ivec)
    model = AutoencoderWithCrop(inputs=[image_input, joint_input],
                                outputs=[decoded_roi, predicted_jvec, roi],
                                name='dplmodel')
    model.summary()
    return model


def show_images(original_images, reconstructed_images, n=10):
    idx = [np.random.randint(1,20) for i in range(n)]
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(3, n, i + 1)
        plt.title("original")
        plt.imshow(original_images[idx[i]])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        cx = plt.subplot(3, n, i + n + 1)
        plt.title("reconstructed")
        plt.imshow(reconstructed_images[idx[i]])
        cx.get_xaxis().set_visible(False)
        cx.get_yaxis().set_visible(False)

    plt.show()


class GeneratorTest:
    def __init__(self):
        self.batch_size = batch_size
        self.time_window = time_window_size
        train_ds = load_dataset(groups=range(1,50))
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(train_ds, const_roi_fun, batch_size=self.batch_size, time_window_size=self.time_window, add_roi=False)
    def batch(self, n=10):
        start = time.time()
        for i in range(n):
            batch = next(self.train_gen)
        end = time.time()
        print('\ntook {}[sec]'.format((end-start)/n))

class ROI_AE_LSTM_Trainer:

    def __init__(self):
        self.batch_size = batch_size
        self.time_window = time_window_size
        self.model = model_roi_ae_lstm()

        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        # create checkpoint and save best weight
        self.checkpoint_path = "/home/ryo/Program/moonshot/ae_lstm/runs/ae_cp/cp.ckpt"

    def load_train_data(self):
        # pushing: group1 - group400
        self.train_ds = load_dataset(groups=range(1,300))
        #train_ds = load_dataset(groups=range(1,50))
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(self.train_ds, const_roi_fun, batch_size=self.batch_size, time_window_size=self.time_window, add_roi=False)

    def load_val_data(self):
        self.val_ds = load_dataset(groups=range(300,350))
        val_generator = DPLGenerator()
        self.val_gen = val_generator.flow(self.val_ds, const_roi_fun, batch_size=self.batch_size, time_window_size=self.time_window, add_roi=False)

    def train(self, epochs=100):
        self.load_train_data()
        self.load_val_data()
        start = time.time()

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         mode='min',
                                                         save_best_only=True)

        # early stopping if not changing for 50 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=100)

        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=50,
                                                         verbose=1,
                                                         min_lr=0.00001)

        profiler = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                  histogram_freq = 1,
                                                  profile_batch = '15,25'
                                                  )

        # train the model
        total_train = self.train_gen.number_of_data()
        total_val = self.val_gen.number_of_data()
        history = self.model.fit(self.train_gen,
                                 steps_per_epoch=total_train // self.batch_size,
                                 epochs=epochs,
                                 validation_data=self.val_gen,
                                 validation_steps=total_val // self.batch_size,
                                 callbacks=[cp_callback, early_stop, reduce_lr, profiler])

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))

    def test(self):
        self.prepare_for_test()
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions, estimated_rois = self.model.predict(x)
        visualize_ds(y[0], estimated_rois)
        visualize_ds(x[0][:,0,:,:,:], estimated_rois)
        visualize_ds(predicted_images)
        plt.show()
        return x[1],y[1],predicted_joint_positions,estimated_rois

    def test_joint(self):
        self.prepare_for_test()
        x,y = next(self.val_gen)
        predicted_images, predicted_joint_positions, estimated_rois = self.model.predict(x)
        data = np.concatenate((x[1], predicted_joint_positions[:,np.newaxis,:]), axis=1)

        fig = plt.figure()
        fig.subplots_adjust(hspace=0.1)
        for joint_id in range(dof):
            ax = fig.add_subplot(8//2, 2, joint_id+1)
            ax.plot(np.transpose(data[:,:,joint_id]))
        #ax.axis('off')
        plt.show()

    def create_anim_gif(self, group_num=0, start=0, length=64):
        seq_len = len(self.val_ds[group_num][1])
        ishape = self.val_ds[group_num][1][0].shape
        jv_dim = self.val_ds[group_num][0].shape[1]
        batch_size = seq_len-self.time_window
        batch_x_imgs = np.empty((batch_size, self.time_window, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size, ishape[0], ishape[1], ishape[2]))
        batch_y_jv = np.empty((batch_size, jv_dim))
        for i, seq_num in enumerate(range(batch_size)):
            batch_x_jvs[i] = self.val_ds[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[i] = self.val_ds[group_num][0][seq_num+self.time_window]
            batch_x_imgs[i] = self.val_ds[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[i] = self.val_ds[group_num][1][seq_num+self.time_window]

        # return (batch_x_imgs, batch_x_jvs), (batch_y_img, batch_y_jv)

        ximgs = batch_x_imgs[start:start+length:2]
        xjvs = batch_x_jvs[start:start+length:2]
        predicted_images, predicted_joint_positions, estimated_rois = self.model.predict((ximgs, xjvs))

        yimgs = batch_y_img[start:start+length:2]
        create_anim_gif_for_group(yimgs, estimated_rois[:,0,:], group_num)

    def process_sequence(self, group_num=0):
        res = []
        seq_len = len(self.val_ds[group_num][1])
        ishape = self.val_ds[group_num][1][0].shape
        jv_dim = self.val_ds[group_num][0].shape[1]
        batch_size = 32
        batch_x_imgs = np.empty((batch_size, self.time_window, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size, ishape[0], ishape[1], ishape[2]))
        batch_y_jv = np.empty((batch_size, jv_dim))

        for seq_num in range(seq_len-self.time_window):
            print(seq_num)
            batch_x_jvs[:] = self.val_ds[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[:] = self.val_ds[group_num][0][seq_num+self.time_window]
            batch_x_imgs[:] = self.val_ds[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[:] = self.val_ds[group_num][1][seq_num+self.time_window]

            predicted_images, predicted_joint_positions, estimated_rois = self.model.predict((batch_x_imgs, batch_x_jvs))
            current_joint_position = copy.copy(batch_x_jvs[0][-1])
            label_joint_position = copy.copy(batch_y_jv[0])
            predicted_joint_position = predicted_joint_positions[0]
            res.append((current_joint_position, predicted_joint_position, label_joint_position))
        return res

    def prepare_for_test(self, load_val_data=True):
        if load_val_data:
            self.load_val_data()
        self.model.compile(loss='mse', optimizer=self.opt)
        self.model.load_weights(self.checkpoint_path)
        
    def generate_sequence(self):
        # load best checkpoint and evaluate
        self.model.compile(loss='mse', optimizer=self.opt)
        self.model.load_weights(self.checkpoint_path)

        reconstructed_images = self.model.predict(self.val_data)

        show_images(roi_images, reconstructed_images)

        return reconstructed_images, roi_images


class LSTM_trainer:

    def __init__(self, dof=8):
        super(JointLSTM, self).__init__()

        self._maxlen = 20
        self._dof = dof

        self.lstm = tf.keras.Sequential([
            #tf.keras.layers.InputLayer(batch_input_shape=(None, self._maxlen, self._dof)),
            tf.keras.layers.InputLayer(input_shape=(None, self._dof)),
            tf.keras.layers.LSTM(self._dof, return_sequences=True), # stacked LSTM
            tf.keras.layers.LSTM(self._dof, return_sequences=True),
            tf.keras.layers.LSTM(50),
            tf.keras.layers.Dense(self._dof)
        ])

    def call(self, x):
        return self.lstm(x)

