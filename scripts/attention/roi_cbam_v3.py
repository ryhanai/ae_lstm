# -*- coding: utf-8 -*-
""" task-context conditioned ROI

* version 2
* The model predicts only attended features instead of the whole image.

"""


import os
import numpy as np
import argparse
import tensorflow as tf

from core.utils import *
from core.model import *
from core import generator
from core import trainer
from attention import attention_map2roi
from attention import AttentionLSTMCell

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

input_image_size = (80, 160)
time_window_size = 5
latent_dim = 64
dof = 7


class WeightedFeaturePredictorModel(tf.keras.Model):
    def __init__(self, image_encoder, *args, **kargs):
        super(WeightedFeaturePredictorModel, self).__init__(*args, **kargs)
        tracker_names = ['image_loss', 'joint_loss', 'ae_loss', 'loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = tf.keras.metrics.Mean(name=n)
            self.test_trackers[n] = tf.keras.metrics.Mean(name='val_'+n)

        self._image_encoder = image_encoder
        self.n_steps = 1

    def train_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.random.uniform(shape=(batch_size, 2), minval=-1, maxval=1)
        # input_noise = tf.zeros(shape=(batch_size, 2))

        with tf.GradientTape() as tape:
            image_loss, joint_loss, ae_loss, loss = self.compute_loss(data, input_noise, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['image_loss'].update_state(image_loss)
        self.train_trackers['joint_loss'].update_state(joint_loss)
        self.train_trackers['ae_loss'].update_state(ae_loss)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        x_image, x_joint = x
        batch_size = tf.shape(x_image)[0]
        input_noise = tf.zeros(shape=(batch_size, 2))

        image_loss, joint_loss, ae_loss, loss = self.compute_loss(data, input_noise, training=False)

        self.test_trackers['image_loss'].update_state(image_loss)
        self.test_trackers['joint_loss'].update_state(joint_loss)
        self.test_trackers['ae_loss'].update_state(ae_loss)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, data, input_noise, training):
        x, y = data
        x_image, x_joint = x

        image_loss = 0.0
        joint_loss = 0.0
        y_images, y_joints = y
        y_images_tr = tf.transpose(y_images, [1, 0, 2, 3, 4])
        y_joints_tr = tf.transpose(y_joints, [1, 0, 2])

        for n in range(self.n_steps):
            pred_image_feature, pred_joint, attention_map, decoded_image = self((x_image, x_joint, input_noise), training=training)  # Forward pass

            if n < self.n_steps - 1:
                pass
                # x_image_tr = tf.transpose(x_image, [1, 0, 2, 3, 4])
                # x_image_tr_list = tf.unstack(x_image_tr)
                # x_image_tr_list[1:].append(pred_image_feature)
                # x_image_tr = tf.stack(x_image_tr_list)
                # x_image = tf.transpose(x_image_tr, [1, 0, 2, 3, 4])
                # x_joint_tr = tf.transpose(x_joint, [1, 0, 2])
                # x_joint_tr_list = tf.unstack(x_joint_tr)
                # x_joint_tr_list[1:].append(pred_joint)
                # x_joint_tr = tf.stack(x_joint_tr_list)
                # x_joint = tf.transpose(x_joint_tr, [1, 0, 2])
            else:
                y_image_aug = translate_image(y_images_tr[n], input_noise)
                y_image_aug = tf.expand_dims(y_image_aug, 0)
                y_image_augs = tf.transpose(tf.tile(y_image_aug, [time_window_size, 1, 1, 1, 1]), [1, 0, 2, 3, 4])
                y_image_features = self._image_encoder(y_image_augs, training=training)
                y_image_feature = tf.transpose(y_image_features, [1, 0, 2, 3, 4])[0]

                # image_loss = tf.reduce_mean(tf.square(attention_map * y_image_feature - pred_image_feature))
                image_loss = tf.reduce_mean(tf.square(y_image_feature - pred_image_feature))
                joint_loss = tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))
                ae_loss = tf.reduce_mean(tf.square(y_image_aug - decoded_image))

        loss = joint_loss + ae_loss
        return image_loss, joint_loss, ae_loss, loss

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


def model_weighted_feature_prediction(input_image_shape, time_window_size, image_vec_dim, dof, image_noise=0.3, joint_noise=0.02, use_color_augmentation=True):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    noise_input = tf.keras.Input(shape=(2,))

    x = image_input

    if use_color_augmentation:
        x = TimeDistributedColorAugmentation(time_window_size=time_window_size)(x)
    x = TimeDistributedGeometricalAugmentation(time_window_size=time_window_size)(x, noise_input)
    x = tf.keras.layers.GaussianNoise(image_noise)(x)

    # convert to feature map
    image_encoder = model_time_distributed_encoder(input_image_shape, time_window_size, name='feature-encoder')
    image_feature = image_encoder(x)

    joint_input_with_noise = tf.keras.layers.GaussianNoise(joint_noise)(joint_input)

    cell = AttentionLSTMCell.AttentionLSTMCell(image_vec_dim + dof)
    layer = tf.keras.layers.RNN(cell)
    x, attention_map = layer((image_feature, joint_input_with_noise))

    predicted_ivec = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(x)
    predicted_jvec = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:], output_shape=(dof,))(x)

    channels = 128
    h = 5
    w = 10
    x = tf.keras.layers.Dense(h*w*channels, activation='relu')(predicted_ivec)
    x = tf.keras.layers.Reshape(target_shape=(h, w, channels))(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = deconv_block(x, channels*2)
    x = deconv_block(x, channels)

    predicted_img_feature = x

    # decode to the next frame
    decoded_img = model_decoder((h*4, w*4, channels))(x)
    
    m = WeightedFeaturePredictorModel(inputs=[image_input, joint_input, noise_input],
                                      outputs=[predicted_img_feature, predicted_jvec, attention_map, decoded_img],
                                      image_encoder=image_encoder,
                                      name='roi_cbam_v3')

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

    def predict(self, x):
        noise = tf.zeros(shape=(x[0].shape[0], 2))
        y_pred = self.model.predict(x+(noise,))
        return y_pred

    def predict_images(self, return_data=False, save_to_file=False, normalize_for_visualization=False):
        x, y = next(self.val_gen)

        noise = tf.zeros(shape=(x[0].shape[0], 2))
        y_pred = self.model.predict(x+(noise,))
        predicted_images, predicted_joints, attention_map = y_pred

        visualize_ds(x[0][:,-1,:,:,:])
        if save_to_file:
            plt.savefig('test_rgb.png')
            plt.close()

        if normalize_for_visualization:
            a = attention_map / np.max(attention_max)
        else:
            a = attention_map
        visualize_ds(a)

        if save_to_file:
            plt.savefig('test_attention.png')
            plt.close()
        else:
            plt.show()
        if return_data:
            return x, y_pred

    def post_process(self, input_images, y_pred, n_sigma=0.5, epsilon=1e-2):
        predicted_images, predicted_joints, attention_maps = y_pred
        b, rect = attention_map2roi.apply_filters(input_images, attention_maps, visualize_result=False, return_result=True, n_sigma=n_sigma, epsilon=epsilon)

        cm = plt.get_cmap('viridis')
        a = tf.squeeze(attention_maps[0])
        a = np.array(list(map(cm, a*3.)))[:, :, :3]
        resized_attention_map = tf.image.resize(a, b[0].shape[:2])
        res = tf.concat([b[0], resized_attention_map], axis=1)
        return res.numpy()


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

            predicted_images, predicted_joints, attention_maps, decoded_images = y_pred
            # attention_maps = attention_maps[:, -1]
            input_img = batch_x_imgs[:, -1]
            b, rect = attention_map2roi.apply_filters(input_img, attention_maps, visualize_result=False, return_result=True, n_sigma=n_sigma, epsilon=epsilon)

            # roi_imgs = tf.image.crop_and_resize(input_img, tf.cast(rect, tf.float32), range(batch_size), input_image_size)
            # res = tf.concat([b[0], roi_imgs[0]], axis=0)

            cm = plt.get_cmap('viridis')
            a = tf.squeeze(attention_maps[0])
            a = np.array(list(map(cm, a)))[:, :, :3]
            # a = np.array(list(map(cm, a*5.)))[:, :, :3]
            resized_attention_map = tf.image.resize(a, b[0].shape[:2])

            if out_roi_image:
                roi_imgs = tf.image.crop_and_resize(input_img, tf.cast(rect, tf.float32), range(batch_size), input_image_size)
                res = tf.concat([b[0], resized_attention_map, roi_imgs[0]], axis=1)
            else:
                res = tf.concat([b[0], resized_attention_map], axis=1)
            results.append(res.numpy())

        create_anim_gif_from_images(results, out_filename='estimated_roi_g{:05d}.gif'.format(group_num))


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_gen, save_dir='.'):
        self._val_gen = val_gen
        self._save_dir = save_dir

    def on_epoch_end(self, epoch, logs=None):
        print('Evaluating Model...')
        x, y = next(self._val_gen)
        noise = tf.zeros(shape=(x[0].shape[0], 2))
        y_pred = self.model.predict(x+(noise,))
        predicted_images, predicted_joints, attention_map, decoded_images = y_pred

        visualize_ds(x[0][:,-1,:,:,:])
        plt.savefig(self._save_dir + '/{:04d}_rgb.png'.format(epoch))
        # plt.close()

        a = attention_map
        a /= (np.max(a) - np.min(a))
        visualize_ds(a)
        plt.savefig(self._save_dir + '/{:04d}_attention.png'.format(epoch))
        # plt.close()


predictor = model_weighted_feature_prediction(input_image_size+(3,), time_window_size, latent_dim, dof, use_color_augmentation=True)


def prepare(task, dataset):
    if task == 'train':
        train_ds = Dataset(dataset, mode='train')
        train_ds.load(image_size=input_image_size)
        train_ds.preprocess(time_window_size)
        val_ds = Dataset(dataset, mode='test')
        val_ds.load(image_size=input_image_size)
        val_ds.preprocess(time_window_size)
        tr = trainer.TimeSequenceTrainer(predictor, train_ds, val_ds, time_window_size=time_window_size)
        return tr
    elif task == 'test':
        if dataset == 'reaching-real' or dataset == 'reaching-real-destructor':
            pass
        elif dataset == 'kitting':
            pass
        elif dataset == 'kitting2':
            pass
        elif dataset == 'kitting3':
            cp = 'ae_cp.kitting3.roi_cbam_v3.20230516151303'
            cp_epoch = 224
        elif dataset == 'pen-kitting-real':
            cp = 'ae_cp.pen-kitting-real.roi_cbam_v3.20230516173339'
            cp_epoch = 272
        elif dataset == 'liquid-pouring':
            pass
        val_ds = Dataset(dataset, mode='test')
        val_ds.load(image_size=input_image_size)
        tr = Tester(predictor, val_ds, checkpoint_file=cp, checkpoint_epoch=cp_epoch)
        return tr


def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-t', '--task', type=str, default='test')  # train | test
    parser.add_argument('-d', '--dataset', type=str, default='pen-kitting-real')  # kitting2 | kitting | reaching-real | reaching-real-destructor | liquid-pouring | pen-kitting-real
    parser.add_argument('-s', '--start_training', action='store_true')
    args = parser.parse_args()
    message('task = {}'.format(args.task))
    message('dataset = {}'.format(args.dataset))
    message('start_training = {}'.format(args.start_training))

    tr = prepare(args.task, args.dataset)
    
    if args.task == 'train':
        cb = CustomCallback(tr.val_gen, save_dir='/'.join(tr.checkpoint_save_path.split('/')[:-1]))
    if args.start_training:
        tr.train(epochs=500, save_best_only=False, early_stop_patience=300, reduce_lr_patience=100, custom_callbacks=[cb])

    return tr


if __name__ == '__main__':
    tr = main()
