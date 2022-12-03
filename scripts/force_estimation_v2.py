# -*- coding: utf-8 -*-

# from typing import Concatenate
import os, time
from datetime import datetime
from pybullet_tools import *
# import force_distribution_viewer
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import res_unet
import tensorflow as tf
# import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from core.utils import *
from force_estimation_data_loader import ForceEstimationDataLoader
from model import *
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.models import Model

dataset = 'basket-filling2'
image_height = 360
image_width = 512
input_image_shape = [image_height, image_width]
num_classes = 62

dl = ForceEstimationDataLoader(
                            os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset),
                            os.path.join(os.environ['HOME'], 'Dataset/dataset2', dataset+'-real'),
                            image_height=image_height,
                            image_width=image_width,
                            start_seq=1,
                            n_seqs=1800,  # n_seqs=1500,
                            start_frame=3, n_frames=3,
                            real_start_frame=1, real_n_frames=294
                            )


def augment(xs, ys):
    img = xs
    fmap = ys

    # color transformation on xs
    # brightness_max_delta=0.2
    # contrast_lower=0.8
    # contrast_upper=1.2
    hue_max_delta = 0.05
    # img = tf.image.random_brightness(img, max_delta=brightness_max_delta)
    # img = tf.image.random_contrast(img, lower=contrast_lower, upper=contrast_upper)
    img = tf.image.random_hue(img, max_delta=hue_max_delta)

    # apply save transform to xs and ys
    batch_sz = tf.shape(xs)[0]
    height = tf.shape(xs)[1]
    # width = tf.shape(xs)[2]
    shift_fmap = 2.0
    shift_height = tf.cast(height * 4 / 40, tf.float32)
    shift_width = tf.cast(height * 4 / 40, tf.float32)
    angle_factor = 0.2
    rnds = tf.random.uniform(shape=(batch_sz, 2))
    img = tfa.image.translate(img, translations=tf.stack([shift_height, shift_width], axis=0)*rnds)
    fmap = tfa.image.translate(fmap, translations=tf.stack([shift_fmap, shift_fmap], axis=0)*rnds)
    rnds = tf.random.uniform(shape=(batch_sz,))
    rnds = rnds - 0.5
    img = tfa.image.rotate(img, angles=angle_factor*rnds)
    fmap = tfa.image.rotate(fmap, angles=angle_factor*rnds)

    # add gaussian noise to xs

    return img, fmap


class ForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(ForceEstimationModel, self).__init__(*args, **kargs)
        # tracker_names = ['loss', 'dloss', 'sloss', 'floss']
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        xs, y_labels = data
        xs, y_labels = augment(xs, y_labels)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True) # Forward pass
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        # self.train_trackers['dloss'].update_state(loss[1])
        # self.train_trackers['sloss'].update_state(loss[2])
        # self.train_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        xs, y_labels = data

        y_pred = self(xs, training=False) # Forward pass
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        # self.test_trackers['dloss'].update_state(loss[1])
        # self.test_trackers['sloss'].update_state(loss[2])
        # self.test_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    # def compute_loss(self, y_labels, y_pred):
    #     '''
    #         compute_loss for rgb to fmap
    #     '''
    #     y_pred_depth, y_pred_logits, y_pred_force = y_pred
    #     depth_labels, seg_labels, force_labels = y_labels
    #     dloss = tf.reduce_mean(tf.square(depth_labels - y_pred_depth))
    #     seg_labels = tf.cast(seg_labels, dtype=tf.int32)
    #     sloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seg_labels, logits=y_pred_logits)
    #     sloss = tf.reduce_mean(sloss)
    #     floss = tf.reduce_mean(tf.square(force_labels - y_pred_force))
    #     loss = dloss + sloss + floss
    #     return loss,dloss,sloss,floss

    def compute_loss(self, y_labels, y_pred):
        '''
            compute_loss for rgb to fmap
        '''
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


class DRCNForceEstimationModel(tf.keras.Model):

    def __init__(self, *args, **kargs):
        super(DRCNForceEstimationModel, self).__init__(*args, **kargs)
        # tracker_names = ['loss', 'dloss', 'sloss', 'floss']
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        src_data, tgt_data = data
        xs, y_labels,  = src_data
        xs, y_labels = augment(xs, y_labels)

        with tf.GradientTape() as tape:
            y_pred = self(xs, training=True) # Forward pass
            loss = self.compute_loss(y_labels, y_pred)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        # self.train_trackers['dloss'].update_state(loss[1])
        # self.train_trackers['sloss'].update_state(loss[2])
        # self.train_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        src_data, tgt_data = data
        xs, y_labels,  = src_data

        y_pred = self(xs, training=False) # Forward pass
        loss = self.compute_loss(y_labels, y_pred)

        self.test_trackers['loss'].update_state(loss)
        # self.test_trackers['dloss'].update_state(loss[1])
        # self.test_trackers['sloss'].update_state(loss[2])
        # self.test_trackers['floss'].update_state(loss[3])
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    # def compute_loss(self, y_labels, y_pred):
    #     '''
    #         compute_loss for rgb to fmap
    #     '''
    #     y_pred_depth, y_pred_logits, y_pred_force = y_pred
    #     depth_labels, seg_labels, force_labels = y_labels
    #     dloss = tf.reduce_mean(tf.square(depth_labels - y_pred_depth))
    #     seg_labels = tf.cast(seg_labels, dtype=tf.int32)
    #     sloss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=seg_labels, logits=y_pred_logits)
    #     sloss = tf.reduce_mean(sloss)
    #     floss = tf.reduce_mean(tf.square(force_labels - y_pred_force))
    #     loss = dloss + sloss + floss
    #     return loss,dloss,sloss,floss

    def compute_loss(self, y_labels, y_pred):
        '''
            compute_loss for rgb to fmap
        '''
        loss = tf.reduce_mean(tf.square(y_labels - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_rgb_to_fmap(input_shape=input_image_shape, 
                      num_filters=[16, 32, 64],
                      kernel_size=3,
                      num_channels=3,
                      num_classes=62,
                      noise_stddev=0.3):
    input = tf.keras.Input(shape=input_shape + [num_channels])

    # Data Augmentation
    x = tf.keras.layers.GaussianNoise(noise_stddev)(input)

    encoder_output = res_unet.encoder(x, num_filters, kernel_size)

    # bridge layer, number of filters is double that of the last encoder layer
    bridge = res_unet.res_block(encoder_output[-1], [num_filters[-1]*2], kernel_size, strides=[2, 1], name='bridge')

    print(encoder_output[-1].shape)
    # decoder_output = res_unet.decoder(bridge, encoder_output, num_filters, kernel_size)
    out1,out2,out3 = res_unet.multi_head_decoder(bridge, encoder_output, num_filters, kernel_size, num_heads=3)

    output_depth = tf.keras.layers.Conv2D(1, 
                                          kernel_size,
                                          strides=1,
                                          padding='same',
                                          name='depth_output')(out1)

    output_seg = tf.keras.layers.Conv2D(num_classes,
                                        kernel_size,
                                        strides=1,
                                        padding='same',
                                        name='seg_output')(out2)

    output_force = tf.keras.layers.Conv2D(20,
                                          kernel_size,
                                          strides=1,
                                          padding='same',
                                          name='force_output')(out3)
    output_force = tf.keras.layers.Resizing(40, 40)(output_force)

    model = ForceEstimationModel(inputs=[input],
                                 outputs=[output_depth, output_seg, output_force],
                                 name='force_estimator')

    model.summary()
    return model


def model_simple_decoder(input_shape, name='decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))

    x = feature_input
    x = deconv_block(x, 1024, with_upsampling=False)  # 12x16
    x = deconv_block(x, 512)                          # 24x32
    x = deconv_block(x, 256, with_upsampling=False)   # 24x32
    x = deconv_block(x, 128, with_upsampling=False)   # 24x32
    x = deconv_block(x, 64)                           # 48x64
    x = deconv_block(x, 32, with_upsampling=False)    # 48x64
    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_resnet_decoder(input_shape, name='resnet_decoder'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [1024, 512], 3, strides=[1, 1], name='resb1')
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (24, 32))
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb3')
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_resnet_decoder2(input_shape, name='resnet_decoder2'):
    feature_input = tf.keras.Input(shape=(input_shape))
    x = feature_input
    x = res_unet.res_block(x, [256, 128], 3, strides=[1, 1], name='resb1')
    x = res_unet.res_block(x, [64, 64], 3, strides=[1, 1], name='resb2')
    x = res_unet.upsample(x, (24, 32))  # (46, 64)
    x = res_unet.res_block(x, [32, 32], 3, strides=[1, 1], name='resb4')

    x = tf.keras.layers.Conv2DTranspose(20, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    x = tf.keras.layers.Resizing(40, 40)(x)
    decoder_output = x
    decoder = keras.Model(feature_input, decoder_output, name=name)
    decoder.summary()
    return decoder


def model_rgb_to_fmap_res50(input_shape=input_image_shape, input_noise_stddev=0.3):
    input_shape = input_shape + [3]
    image_input = tf.keras.Input(shape=input_shape)

    x = image_input

    # augmentation layers
    x = tf.keras.layers.RandomZoom(0.05)(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2, value_range=(0, 1.0))(x)
    x = tf.keras.layers.RandomContrast(factor=0.3)(x)
    x = tf.keras.layers.GaussianNoise(input_noise_stddev)(x)

    resnet50 = ResNet50(include_top=False, input_shape=input_shape)
    encoded_img = resnet50(x)
    decoded_img = model_resnet_decoder((12, 16, 2048))(encoded_img)

    model = ForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    # model = DRCNForceEstimationModel(inputs=[image_input], outputs=[decoded_img], name='model_resnet')
    model.summary()

    return model


def model_depth_to_fmap(input_shape=input_image_shape, kernel_size=3, num_classes=num_classes):
    input_depth = tf.keras.Input(shape=input_shape + [1])
    input_seg = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Lambda(lambda x: tf.one_hot(tf.cast(x, tf.int32), depth=num_classes))(input_seg)
    x = tf.keras.layers.Concatenate(axis=-1)([input_depth, x])

    encoder_output = res_unet.encoder(x, num_filters=(32, 64, 64, 128, 256), kernel_size=kernel_size)
    decoded_img = model_resnet_decoder2((23, 32, 256))(encoder_output[-1])

    model = Model(inputs=[input_depth, input_seg], outputs=[decoded_img], name='model_depth_to_fmap')
    model.summary()

    return model


class Trainer:
    def __init__(self, model,
                 train_dataset,
                 val_dataset,
                 # load_weight=False,
                 batch_size=32,
                 runs_directory=None,
                 checkpoint_file=None):

        self.input_image_shape = val_dataset[0][0].shape
        self.batch_size = batch_size

        self.model = model
        self.opt = keras.optimizers.Adamax(learning_rate=0.001)
        self.model.compile(loss='mse', optimizer=self.opt)

        if train_dataset:
            self.train_ds = train_dataset

        if val_dataset:
            self.val_ds = val_dataset
            self.val_data_loaded = True

        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.getcwd()), 'runs')
        f = checkpoint_file if checkpoint_file else 'ae_cp.{}.{}.{}'.format(dataset, self.model.name, datetime.now().strftime('%Y%m%d%H%M%S'))
        self.checkpoint_path = os.path.join(d, f, 'cp.ckpt')

        if checkpoint_file:
            print('load weights from ', checkpoint_file)
            self.model.load_weights(self.checkpoint_path)

    def prepare_callbacks(self, early_stop_patience=100, reduce_lr_patience=50):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1,
                                                         mode='min',
                                                         save_best_only=True)

        # early stopping if not changing for 50 epochs
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=early_stop_patience)

        # reduce learning rate
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                                         factor=0.1,
                                                         patience=reduce_lr_patience,
                                                         verbose=1,
                                                         min_lr=0.00001)

        profiler = tf.keras.callbacks.TensorBoard(log_dir='logs',
                                                  histogram_freq=1,
                                                  profile_batch='15,25')
        return cp_callback, early_stop, reduce_lr, profiler

    def train(self, epochs=300, early_stop_patience=100, reduce_lr_patience=50):
        xs = self.train_ds[0].astype('float32')
        ys = self.train_ds[1].astype('float32')
        val_xs = self.val_ds[0].astype('float32')
        val_ys = self.val_ds[1].astype('float32')

        start = time.time()
        callbacks = self.prepare_callbacks(early_stop_patience, reduce_lr_patience)

        source_dataset = tf.data.Dataset.from_tensor_slices((xs, ys)).shuffle(len(xs)).batch(self.batch_size)
        # target_dataset = tf.data.Dataset.from_tensor_slices(xs_real).shuffle(len(x_real)).repeat().batch(self.batch_size)
        # train_dataset = tf.data.Dataset.zip((source_dataset, target_dataset))

        val_source_dataset = tf.data.Dataset.from_tensor_slices((val_xs, val_ys)).batch(self.batch_size)
        # # val_target_dataset = tf.data.Dataset.from_tensor_slices(val_xs).shuffle(BUFFER_SIZE).repeat().batch(self.batch_size)
        # val_dataset = tf.data.Dataset.zip((val_source_dataset, val_target_dataset))

        history = self.model.fit(source_dataset,
                                 batch_size=self.batch_size,
                                 epochs=epochs,
                                 callbacks=callbacks,
                                 validation_data=val_source_dataset,
                                 shuffle=True)

        # history = self.model.fit(xs, ys,
        #                          batch_size=self.batch_size,
        #                          epochs=epochs,
        #                          callbacks=callbacks,
        #                          validation_data=(val_xs, val_ys),
        #                          shuffle=True)

        end = time.time()
        print('\ntotal time spent for training: {}[min]'.format((end-start)/60))


def visualize_forcemaps(force_distribution, title='', zaxis_first=False):
    f = force_distribution / np.max(force_distribution)
    fig = plt.figure(figsize=(16, 4))
    fig.subplots_adjust(hspace=0.1)
    fig.suptitle(title, fontsize=28)

    if zaxis_first:
        channels = f.shape[0]
    else:
        channels = f.shape[-1]
    for p in range(channels):
        ax = fig.add_subplot(channels//10, 10, p+1)
        ax.axis('off')
        if zaxis_first:
            ax.imshow(f[p], cmap='gray', vmin=0, vmax=1.0)
        else:
            ax.imshow(f[:, :, p], cmap='gray', vmin=0, vmax=1.0)


def visualize_result(f_prediction, f_label, rgb, filename=None):
    visualize_forcemaps(f_prediction, title='prediction')
    plt.savefig('prediction.png')
    p = plt.imread('prediction.png')[:, :, :3]
    if f_label is None:
        p = cv2.resize(p, (1440,360))
        res = np.concatenate([rgb, p], axis=1)
    if f_label is not None:
        visualize_forcemaps(f_label, title='ground truth')
        plt.savefig('ground_truth.png')
        g = plt.imread('ground_truth.png')[:, :, :3]
        pg = np.concatenate([p, g], axis=0)
        rgb2 = np.ones((800, 512, 3))
        rgb2[220:580] = rgb
        res = np.concatenate([rgb2, pg], axis=1)
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    plt.imshow(res)
    if filename is not None:
        plt.savefig(filename)
    plt.show()


class Tester:
    def __init__(self, model, test_data, checkpoint_file, runs_directory=None):
        self.model = model
        d = runs_directory if runs_directory else os.path.join(os.path.dirname(os.getcwd()), 'runs')
        self.checkpoint_path = os.path.join(d, checkpoint_file, 'cp.ckpt')
        print('load weights from ', self.checkpoint_path)
        self.model.load_weights(self.checkpoint_path)
        self.test_data = test_data

    def predict(self, n):
        xs = self.test_data[0][n:n+1]
        ys = self.test_data[1][n:n+1]
        y_pred = self.model.predict(xs)
        plt.imshow(xs[0])
        visualize_forcemaps(y_pred[0])
        visualize_forcemaps(ys[0])
        plt.show()

    def predict_segmentation_mask(self, n):
        xs = self.test_data[0][n:n+1]
        ys = self.test_data[1][n:n+1]
        y_pred_logits = self.model.predict(xs)
        plt.figure()
        plt.imshow(xs[0])
        y_pred = tf.cast(tf.argmax(y_pred_logits, axis=-1), dtype=tf.uint8)
        plt.figure()
        plt.imshow(y_pred[0])
        plt.figure()
        plt.imshow(ys[0])
        plt.show()
        return xs[0], ys[0], y_pred[0]

    def predict_force_from_rgb(self, n, visualize=True):
        if type(self.test_data) is tuple:
            xs = self.test_data[0][n:n+1]
            force_label = self.test_data[1][n]
        else:
            xs = self.test_data[n:n+1]
            force_label = None
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        visualize_result(y_pred_forces[0], force_label, xs[0], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, xs[0]

    def predict_force_from_rgb_with_img(self, rgb):
        xs = np.expand_dims(rgb, axis=0)
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        visualize_forcemaps(y_pred_forces[0])
        plt.show()
        return y_pred_forces

    def predict_force_from_depth_seg(self, n, rgb):
        xs_depth = self.test_data[0][0][n:n+1]
        xs_seg = self.test_data[0][1][n:n+1]
        xs = xs_depth, xs_seg
        y_preds = self.model.predict(xs)
        y_pred_forces = y_preds
        force_label = self.test_data[1][n]

        visualize_result(y_pred_forces[0], force_label, rgb[n], 'result{:05d}.png'.format(n))
        return y_pred_forces[0], force_label, rgb[n]


# viewer = force_distribution_viewer.ForceDistributionViewer.get_instance()


def show_bin_state(fcam, bin_state, fmap, draw_fmap=True, draw_force_gradient=False):
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = fmap
    positions = fcam.positions
    viewer.publish_bin_state(bin_state, positions, fv, draw_fmap=draw_fmap, draw_force_gradient=draw_force_gradient)


def pick_direction_plan(fcam, n=25, gp=[0.02, -0.04, 0.79], radius=0.05, scale=[0.005, 0.01, 0.004]):
    fmap, force_label, rgb = tester.predict_force_from_rgb(n, visualize_result=False)
    bin_state = tester.test_data[2][n]

    gp = np.array(gp)
    fv = np.zeros((40, 40, 40))
    fv[:, :, :20] = fmap
    gxyz = np.gradient(-fv)
    g_vecs = np.column_stack([g.flatten() for g in gxyz])

    ps = fcam.positions
    idx = np.where(np.sum((ps - gp)*g_vecs, axis=1) < 0)[0]
    fps = ps[idx]
    fg_vecs = g_vecs[idx]

    # points for visualization
    filtered_pos_val_pairs = [(p, g) for (p, g) in zip(fps, fg_vecs) if scipy.linalg.norm(g) > 0.008]
    pz, vz = zip(*filtered_pos_val_pairs)
    pz = np.array(pz)
    vz = np.array(vz)
    viewer.publish_bin_state(bin_state, ps, fv, draw_fmap=False, draw_force_gradient=False)
    viewer.draw_vector_field(pz, vz, scale=0.3)
    viewer.rviz_client.draw_sphere(gp, [1, 0, 0, 1], [0.01, 0.01, 0.01])
    viewer.rviz_client.show()

    # points for planning
    pz = fps
    vz = fg_vecs
    idx = np.where(scipy.linalg.norm(pz - gp, axis=1) < radius)[0]
    pz = pz[idx]
    vz = vz[idx]
    pick_direction = np.sum(vz, axis=0)
    pick_direction /= np.linalg.norm(pick_direction)
    viewer.rviz_client.draw_arrow(gp, gp + pick_direction * 0.1, [0, 1, 0, 1], scale)
    pick_moment = np.sum(np.cross(pz - gp, vz), axis=0)
    pick_moment /= np.linalg.norm(pick_moment)
    viewer.rviz_client.draw_arrow(gp, gp + pick_moment * 0.1, [1, 1, 0, 1], scale)

    viewer.rviz_client.show()
    return pz, vz, pick_direction, pick_moment


def virtual_pick(bin_state0, pick_vector, pick_moment, object_name='011_banana', alpha=0.01, beta=0.05, repeat=5):
    def do_virtual_pick():
        bin_state = bin_state0
        for i in range(10):
            p, q = [s for s in bin_state if s[0] == object_name][0][1]
            dq = quat_from_euler(beta * pick_moment)
            dp = alpha * pick_vector
            p2, q2 = multiply_transforms(dp, dq, p, q)
            p2 = p + dp
            bin_state = [(object_name, (p2, q2)) if s[0] == object_name else s for s in bin_state]
            viewer.publish_bin_state(bin_state, [], [], draw_fmap=False, draw_force_gradient=False)
            time.sleep(0.2)

    for i in range(repeat):
        do_virtual_pick()


task = 'test-real'
# model_file = 'ae_cp.basket-filling.model_resnet.20221115193122' # current best
# model_file = 'ae_cp.basket-filling.model_resnet.20221125134301'
model_file = 'ae_cp.basket-filling2.model_resnet.20221202165608'

if task == 'train':
    model = model_rgb_to_fmap_res50()
    train_data, valid_data = dl.load_data_for_rgb2fmap(train_mode=True)
    trainer = Trainer(model, train_data, valid_data)
elif task == 'adaptation':
    model = model_rgb_to_fmap_res50()
    train_data, valid_data = dl.load_data_for_rgb2fmap_with_real(train_mode=True)
    trainer = Trainer(model, train_data, valid_data)
elif task == 'test':
    model = model_rgb_to_fmap_res50()
    test_data = dl.load_data_for_rgb2fmap(test_mode=True)
    tester = Tester(model, test_data, model_file)
elif task == 'test-real':
    model = model_rgb_to_fmap_res50()
    test_data = dl.load_real_data_for_rgb2fmap(test_mode=True)
    tester = Tester(model, test_data, model_file)
elif task == 'pick':
    model = model_rgb_to_fmap_res50()
    test_data = dl.load_data_for_rgb2fmap(test_mode=True, load_bin_state=True)
    tester = Tester(model, test_data, model_file)

# tester = Tester(model, test_data, 'ae_cp.basket-filling.force_estimator.best')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.force_estimator.20221028172410')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221101213121')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_depth_to_fmap.20221104234049')

# best
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_depth_to_fmap.20221107110203')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221107144923')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221115154455')
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221115182656')

# gaussian / color augmentation
# tester = Tester(model, test_data, 'ae_cp.basket-filling.model_resnet.20221108181626')
