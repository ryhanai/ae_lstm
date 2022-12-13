# -*- coding: utf-8 -*-

import os

import attention_map2roi
import generator
import matplotlib.ticker as ptick
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
import trainer
from core.utils import *
from model import *
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from AttentionLSTMCell import *
from mpl_toolkits.mplot3d import Axes3D, axes3d

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# dataset = 'kitting-destructor'
# train_groups = range(0, 6)
# val_groups = range(0, 6)
# joint_range_data = range(0, 6)

# dataset = 'kitting2'
# train_groups=range(0,9)
# val_groups=range(9,15)

dataset = 'reaching-real'
train_groups=range(0,136)
val_groups=range(136,156)
joint_range_data=range(0,156)

# dataset = 'kitting'
# train_groups=range(0,90)
# val_groups=range(90,111)
# joint_range_data=range(0,111)

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
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

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

    encoder = keras.Model([image_input], x, name=name)
    encoder.summary()
    return encoder


def model_decoder(input_shape, name='decoder'):
    input_feature = tf.keras.Input(shape=(input_shape))
    x = deconv_block(input_feature, 64, with_upsampling=False)
    x = deconv_block(x, 32)
    x = deconv_block(x, 16)
    decoded_img = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)

    decoder = keras.Model([input_feature], decoded_img, name=name)
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

    cell = AttentionLSTMCell(image_vec_dim + dof)
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


wf_predictor = model_weighted_feature_prediction(input_image_size+(3,), time_window_size, latent_dim, dof, use_color_augmentation=True)


def train():
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = trainer.TimeSequenceTrainer(wf_predictor, train_ds, val_ds, time_window_size=time_window_size)
    tr.train(epochs=800, save_best_only=False, early_stop_patience=800, reduce_lr_patience=100)


class Predictor(trainer.TimeSequenceTrainer):
    def __init__(self, *args, time_window_size=20, **kargs):
        super(Predictor, self).__init__(*args, time_window_size=time_window_size, **kargs)

    def get_data(self):
        return next(self.val_gen)

    def get_a_sample_from_batch(self, batch, n):
        x, y = batch
        x_img = x[0][n:n+1]
        x_joint = x[1][n:n+1]
        y_img = y[0][n:n+1]
        y_joint = y[1][n:n+1]
        return (x_img, x_joint), (y_img, y_joint)

    # evaluation of predictor
    def predict_with_roi(self, batch, roi_params):
        x, y = batch
        batch_sz = x[1].shape[0]
        if roi_params.ndim == 1:
            roi_params = np.tile(roi_params, (batch_sz, 1))
        predicted_images, predicted_joints = self.model.predict(x + (roi_params,))
        # visualize_ds(y[0], roi_params) # to draw ROI roi_params must be converted to rectangle

        if y[0].ndim == 5:
            bboxes = roi_rect1((roi_params[:, :2], roi_params[:, 2]))
            imgs = draw_bounding_boxes(x[0][:, -1], bboxes)
            visualize_ds(imgs)
        else:
            visualize_ds(y[0])
        visualize_ds(predicted_images)
        plt.show()

    def generate_roi_images(self, sample, n=5):
        xs = np.linspace(0.1, 0.9, n)
        ys = np.linspace(0.1, 0.9, n)
        ss = 0.7 * np.sin(np.pi*xs)
        out_images = []
        for roi_params in zip(xs, ys, ss):
            roi_params = np.expand_dims(np.array(roi_params), 0)
            predicted_images, _ = self.model.predict(sample[0] + (roi_params,))
            imgs = sample[0][0][0][-1:]
            bboxes = roi_rect1((roi_params[:, :2], roi_params[:, 2]))
            img_with_bb = draw_bounding_boxes(imgs, bboxes)[0]
            img = np.concatenate([img_with_bb, predicted_images[0]], axis=1)
            out_images.append(img)
            # plt.imshow(img)
            # plt.show()
        create_anim_gif_from_images(out_images, 'generated_roi_images.gif')

    def prediction_error(self, sample, roi_pos=[0.5, 0.5], roi_scale=0.8):
        x, y = sample
        batch_size = tf.shape(x[0])[0]
        roi_pos = tf.tile([roi_pos], (batch_size, 1))
        roi_scale = tf.repeat(roi_scale, batch_size)
        image_loss, joint_loss, loss = self.model.do_predict(x, y, roi_pos, roi_scale, training=False)
        return image_loss, joint_loss, loss

    def prediction_errors(self, sample, nx=10, ny=10, ns=1, smin=0.6, smax=1.0):
        x = np.linspace(0.0, 1.0, nx, dtype='float32')
        y = np.linspace(0.0, 1.0, ny, dtype='float32')
        s = np.linspace(smin, smax, ns, dtype='float32')
        x, y, s = np.meshgrid(x, y, s)
        z = np.array([self.prediction_error(sample, [x, y], s)[1] for x, y, s in zip(x.flatten(), y.flatten(), s.flatten())]).reshape((nx, ny, ns))
        idx = np.unravel_index(np.argmin(z), z.shape)
        xopt = x[idx]
        yopt = y[idx]
        sopt = s[idx]
        sopt_idx = idx[2]
        imgs = sample[0][0][0][-1:]
        bboxes = roi_rect1((np.array([[xopt,yopt]]), np.array([sopt])))
        img_with_bb = draw_bounding_boxes(imgs, bboxes)[0]
        plt.imshow(img_with_bb)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel('v')
        ax.set_ylabel('u')
        ax.zaxis.set_major_formatter(ptick.ScalarFormatter(useMathText=True))
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0, 0))
        ax.plot_surface(x[:, :, sopt_idx], y[:, :, sopt_idx], z[:, :, sopt_idx], cmap='plasma')

        plt.show()
        # return x,y,z,imgs[0]

    # evaluation of ROI estimator
    def predict_images(self, batch=None):
        if batch is None:
            x, y = next(self.val_gen)
        else:
            x, y = batch
        batch_size = tf.shape(x[0])[0]

        roi_params = np.tile([0.5, 0.5, 0.8], (batch_size, 1))
        roi_params = self.model.predict(x+(roi_params,))
        bboxes = roi_rect1((roi_params[:, :2], roi_params[:, 2]))

        imgs = draw_bounding_boxes(x[0][:, -1], bboxes)
        visualize_ds(imgs)
        # predicted_images, predicted_joints = self.model.predictor.predict(x + (roi_params,))
        # visualize_ds(predicted_images)
        plt.show()

    # def predict(self, x):
    #     batch_sz = x[1].shape[0]
    #     roi_params0 = np.tile([0.5,0.5,0.8], (batch_sz,1))
    #     roi_params = self.model.predict(x+(roi_params0,))
    #     #pred_img, pred_joint = self.model.predictor.predict(x + (roi_params,))
    #     #pred_img, pred_joint = self.model.predictor.predict(x + (roi_params0,))
    #     #bboxes = roi_rect1((roi_params[:,:2], roi_params[:,2]))
    #     return pred_img, pred_joint, bboxes

    def predict(self, x):
        'only used for Predictor test'
        batch_sz = x[1].shape[0]
        roi_params0 = np.tile([0.5, 0.5, 0.8], (batch_sz, 1))
        pred_img, pred_joint = self.model.predict(x+(roi_params0,))
        bboxes = roi_rect1((roi_params0[:, :2], roi_params0[:, 2]))
        return pred_img, pred_joint, bboxes

    def get_validation_sample(self, group_num=0, start_point=0):
        d = self.val_ds.data
        batch_x_jv = np.expand_dims(d[group_num][0][start_point:start_point+self.time_window], axis=0)
        batch_y_jv = np.expand_dims(d[group_num][0][start_point+self.time_window:start_point+self.time_window+self.val_gen.prediction_window_size], axis=0)
        batch_x_img = np.expand_dims(np.array(d[group_num][1][start_point:start_point+self.time_window]), axis=0)
        batch_y_img = np.expand_dims(np.array(d[group_num][1][start_point+self.time_window:start_point+self.time_window+self.val_gen.prediction_window_size]), axis=0)
        return (batch_x_img, batch_x_jv), (batch_y_img, batch_y_jv)

    def predict_for_group(self, group_num=0):
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

        roi_params = np.tile([0.5,0.5,0.8], (batch_size,1))
        results = []

        for seq_num in range(seq_len-self.time_window):
            print(seq_num)
            batch_x_jvs[:] = d[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[:] = d[group_num][0][seq_num+self.time_window]
            batch_x_imgs[:] = d[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[:] = d[group_num][1][seq_num+self.time_window]
            estimated_roi_params = self.model.predict((batch_x_imgs, batch_x_jvs, roi_params))
            bboxes = roi_rect1((estimated_roi_params[:,:2], estimated_roi_params[:,2]))
            imgs = draw_bounding_boxes(batch_x_imgs[:,-1], bboxes)
            results.append(imgs[0].numpy())

        create_anim_gif_from_images(results, out_filename='estimated_roi_g{:05d}.gif'.format(group_num))


class Tester(trainer.Trainer):

    def __init__(self, 
                 model,
                 val_dataset,
                 time_window_size=time_window_size,
                 batch_size=32,
                 runs_directory=None,
                 checkpoint_file=None):

        super(Tester, self).__init__(model,
                                     None,
                                     val_dataset,
                                     batch_size=batch_size,
                                     runs_directory=runs_directory,
                                     checkpoint_file=checkpoint_file)

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


def prepare_for_test(cp='ae_cp.kitting.weighted_feature_prediction.20220907161250'):
    # ae_cp.reaching-real.weighted_feature_prediction.20220623184031
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = Tester(wf_predictor, val_ds, checkpoint_file=cp)
    return tr

def prepare_for_test_destructor(cp='ae_cp.reaching-real.weighted_feature_prediction.20220628160446'):
    # ae_cp.reaching-real.weighted_feature_prediction.20220623184031
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=destructor_groups, image_size=input_image_size)
    tr = Tester(wf_predictor, val_ds, checkpoint_file=cp)
    return tr


# if __name__ == "__main__":
#    train()
