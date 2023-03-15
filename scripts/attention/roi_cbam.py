# -*- coding: utf-8 -*-

import os
import numpy as np
import argparse
import tensorflow as tf

from core.utils import *
from core.model import *
from core import generator
from core import trainer
import attention_map2roi
import AttentionLSTMCell

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

input_image_size = (80, 160)
time_window_size = 20
latent_dim = 64
dof = 7


class WeightedFeaturePredictorModel(tf.keras.Model):
    def __init__(self, *args, **kargs):
        super(WeightedFeaturePredictorModel, self).__init__(*args, **kargs)
        tracker_names = ['image_loss', 'joint_loss', 'loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = tf.keras.metrics.Mean(name=n)
            self.test_trackers[n] = tf.keras.metrics.Mean(name='val_'+n)

        self.n_steps = 3

    def train_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.random.uniform(shape=(batch_size, 2), minval=-1, maxval=1)

        image_loss = 0.0
        joint_loss = 0.0
        y_images, y_joints = y
        y_images_tr = tf.transpose(y_images, [1, 0, 2, 3, 4])
        y_joints_tr = tf.transpose(y_joints, [1, 0, 2])

        with tf.GradientTape() as tape:
            for n in range(self.n_steps):
                pred_image, pred_joint, attention_map = self((x_image, x_joint, input_noise), training=True) # Forward pass

                y_image_aug = translate_image(y_images_tr[n], input_noise)
                image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
                joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

                if n < self.n_steps:
                    x_image_tr = tf.transpose(x_image, [1, 0, 2, 3, 4])
                    x_image_tr_list = tf.unstack(x_image_tr)
                    x_image_tr_list[1:].append(pred_image)
                    x_image_tr = tf.stack(x_image_tr_list)
                    x_image = tf.transpose(x_image_tr, [1, 0, 2, 3, 4])
                    x_joint_tr = tf.transpose(x_joint, [1, 0, 2])
                    x_joint_tr_list = tf.unstack(x_joint_tr)
                    x_joint_tr_list[1:].append(pred_joint)
                    x_joint_tr = tf.stack(x_joint_tr_list)
                    x_joint = tf.transpose(x_joint_tr, [1, 0, 2])

            image_loss /= self.n_steps
            joint_loss /= self.n_steps
            loss = image_loss + joint_loss

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['image_loss'].update_state(image_loss)
        self.train_trackers['joint_loss'].update_state(joint_loss)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.zeros(shape=(batch_size, 2))

        image_loss = 0.0
        joint_loss = 0.0
        y_images, y_joints = y

        y_images_tr = tf.transpose(y_images, [1, 0, 2, 3, 4])
        y_joints_tr = tf.transpose(y_joints, [1, 0, 2])

        for n in range(self.n_steps):
            pred_image, pred_joint, attention_map = self((x_image, x_joint, input_noise), training=False)  # Forward pass

            y_image_aug = translate_image(y_images_tr[n], input_noise)
            image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
            joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

            if n < self.n_steps:
                x_image_tr = tf.transpose(x_image, [1, 0, 2, 3, 4])
                x_image_tr_list = tf.unstack(x_image_tr)
                x_image_tr_list[1:].append(pred_image)
                x_image_tr = tf.stack(x_image_tr_list)
                x_image = tf.transpose(x_image_tr, [1, 0, 2, 3, 4])
                x_joint_tr = tf.transpose(x_joint, [1, 0, 2])
                x_joint_tr_list = tf.unstack(x_joint_tr)
                x_joint_tr_list[1:].append(pred_joint)
                x_joint_tr = tf.stack(x_joint_tr_list)
                x_joint = tf.transpose(x_joint_tr, [1, 0, 2])

        image_loss /= self.n_steps
        joint_loss /= self.n_steps
        loss = image_loss + joint_loss

        self.test_trackers['image_loss'].update_state(image_loss)
        self.test_trackers['joint_loss'].update_state(joint_loss)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def model_time_distributed_encoder(input_shape, time_window_size, name='encoder'):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_shape))

    x = time_distributed_conv_block(image_input, 16)
    x = time_distributed_conv_block(x, 32)
    x = time_distributed_conv_block(x, 64, with_pooling=False)
    x = time_distributed_conv_block(x, 128, with_pooling=False)

    encoder = tf.keras.Model([image_input], x, name=name)
    encoder.summary()
    return encoder


def model_decoder(input_shape, name='decoder'):
    input_feature = tf.keras.Input(shape=(input_shape))
    x = deconv_block(input_feature, 64, with_upsampling=False)
    x = deconv_block(x, 32)
    x = deconv_block(x, 16)
    decoded_img = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = tf.keras.Model([input_feature], decoded_img, name=name)
    decoder.summary()
    return decoder


def model_weighted_feature_prediction(input_image_shape, time_window_size, image_vec_dim, dof, image_noise=0.2, joint_noise=0.02, use_color_augmentation=False):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    noise_input = tf.keras.Input(shape=(2,))

    x = image_input

    # This should be changed to time-series version
    if use_color_augmentation:
        x = TimeDistributedColorAugmentation()(x)
    x = TimeDistributedGeometricalAugmentation()(x, noise_input)
    x = tf.keras.layers.GaussianNoise(image_noise)(x)

    # convert to feature map
    image_feature = model_time_distributed_encoder(input_image_shape, time_window_size, name='feature-encoder')(x)

    joint_input_with_noise = tf.keras.layers.GaussianNoise(joint_noise)(joint_input)

    cell = AttentionLSTMCell.AttentionLSTMCell(image_vec_dim + dof)
    layer = tf.keras.layers.RNN(cell)
    x, attention_map = layer((image_feature, joint_input_with_noise))

    predicted_ivec = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(x)
    predicted_jvec = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:], output_shape=(dof,))(x)

    channels = 64
    h = 5
    w = 10
    x = tf.keras.layers.Dense(h*w*channels, activation='relu')(predicted_ivec)
    x = tf.keras.layers.Reshape(target_shape=(h, w, channels))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = deconv_block(x, 128)
    x = deconv_block(x, channels)

    # decode to the next frame
    predicted_img = model_decoder((h*4, w*4, channels))(x)

    m = WeightedFeaturePredictorModel(inputs=[image_input, joint_input, noise_input],
                                      outputs=[predicted_img, predicted_jvec, attention_map],
                                      name='weighted_feature_prediction')

    m.summary()
    return m


class Tester(trainer.Trainer):

    def __init__(self, 
                 model,
                 val_dataset,
                 time_window_size=time_window_size,
                 batch_size=32,
                 runs_directory=None,
                 checkpoint_file=None,
                 checkpoint_epoch=100):

        super(Tester, self).__init__(model,
                                     None,
                                     val_dataset,
                                     batch_size=batch_size,
                                     runs_directory=runs_directory,
                                     checkpoint_file=checkpoint_file,
                                     checkpoint_epoch=checkpoint_epoch)

        self.time_window = time_window_size

        if val_dataset:
            self.val_gen = generator.DPLGenerator().flow(val_dataset.get(),
                                                         None,
                                                         batch_size=self.batch_size,
                                                         time_window_size=self.time_window,
                                                         prediction_window_size=5,
                                                         add_roi=False)
            self.val_data_loaded = True

    def predict_images(self, random_shift=True, return_data=False):
        x, y = next(self.val_gen)

        if random_shift:
            noise = tf.zeros(shape=(x[0].shape[0], 2))
            y_pred = self.model.predict(x+(noise,))
        else:
            y_pred = self.model.predict(x)

        predicted_images, predicted_joints, attention_map = y_pred

        visualize_ds(x[0][:,-1,:,:,:])
        visualize_ds(predicted_images)

        # a = attention_map[:,-1]
        a = attention_map
        a /= (np.max(a) - np.min(a))
        visualize_ds(a)
        plt.show()
        if return_data:
            return x, y_pred

    def predict_for_group(self, group_num=0, random_shift=True, out_roi_image=False, n_sigma=1.5, epsilon=1e-2):
        res = []
        d = self.val_ds.data
        seq_len = len(d[group_num][1])
        ishape = d[group_num][1][0].shape
        jv_dim = d[group_num][0].shape[1]
        batch_size = 1
        batch_x_imgs = np.empty((batch_size, self.time_window) + ishape)
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size,) + ishape)
        batch_y_jv = np.empty((batch_size, jv_dim))

        results = []

        for seq_num in range(seq_len-self.time_window):
            print(seq_num)
            batch_x_jvs[:] = d[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[:] = d[group_num][0][seq_num+self.time_window]
            batch_x_imgs[:] = d[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[:] = d[group_num][1][seq_num+self.time_window]

            if random_shift:
                noise = tf.zeros(shape=(batch_size, 2))
                y_pred = self.model.predict((batch_x_imgs, batch_x_jvs, noise))
            else:
                y_pred = self.model.predict((batch_x_imgs, batch_x_jvs))

            predicted_images, predicted_joints, attention_maps = y_pred
            # attention_maps = attention_maps[:, -1]
            input_img = batch_x_imgs[:, -1]
            b, rect = attention_map2roi.apply_filters(input_img, attention_maps, visualize_result=False, return_result=True, n_sigma=n_sigma, epsilon=epsilon)

            # roi_imgs = tf.image.crop_and_resize(input_img, tf.cast(rect, tf.float32), range(batch_size), input_image_size)
            # res = tf.concat([b[0], roi_imgs[0]], axis=0)

            cm = plt.get_cmap('viridis')
            a = tf.squeeze(attention_maps[0])
            a = np.array(list(map(cm, a*5.)))[:, :, :3]
            resized_attention_map = tf.image.resize(a, b[0].shape[:2])

            if out_roi_image:
                roi_imgs = tf.image.crop_and_resize(input_img, tf.cast(rect, tf.float32), range(batch_size), input_image_size)
                res = tf.concat([b[0], resized_attention_map, roi_imgs[0]], axis=1)
            else:
                res = tf.concat([b[0], resized_attention_map], axis=1)
            results.append(res.numpy())

        create_anim_gif_from_images(results, out_filename='estimated_roi_g{:05d}.gif'.format(group_num))



parser = argparse.ArgumentParser(description='')
parser.add_argument('-t', '--task', type=str, default='test')  # train | test
parser.add_argument('-d', '--dataset', type=str, default='reaching-real')  # kitting | reaching-real | reaching-real-destructor | liquid-pouring
parser.add_argument('-s', '--start_training', action='store_true')
args = parser.parse_args()
message('task = {}'.format(args.task))
message('dataset = {}'.format(args.dataset))
message('start_training = {}'.format(args.start_training))


wf_predictor = model_weighted_feature_prediction(input_image_size+(3,), time_window_size, latent_dim, dof, use_color_augmentation=True)


if args.task == 'train':
    train_ds = Dataset(args.dataset, mode='train')
    train_ds.load(image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(args.dataset, mode='test')
    val_ds.load(image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.TimeSequenceTrainer(wf_predictor, train_ds, val_ds, time_window_size=time_window_size)
    if args.start_training:
        tr.train(epochs=800, save_best_only=False, early_stop_patience=800, reduce_lr_patience=100)
elif args.task == 'test':
    if args.dataset == 'reaching-real' or args.dataset == 'reaching-real-destructor':
        cp = 'ae_cp.reaching-real.weighted_feature_prediction.20221213011838'
        cp_epoch = None
        # cp_epoch = 192
    elif args.dataset == 'kitting':
        cp = 'ae_cp.kitting.weighted_feature_prediction.20221210012310'
        cp_epoch = None
    elif args.dataset == 'liquid-pouring':
        cp = None
        cp_epoch = None
    val_ds = Dataset(args.dataset, mode='test')
    val_ds.load(image_size=input_image_size)
    tr = Tester(wf_predictor, val_ds, checkpoint_file=cp, checkpoint_epoch=cp_epoch)
