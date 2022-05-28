# -*- coding: utf-8 -*-

import os
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
import tensorflow_addons as tfa

from core.utils import *
#from core.model import *
from model import *
import trainer


import matplotlib.ticker as ptick

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'


dataset = 'reaching-real'
train_groups=range(0,136)
val_groups=range(136,156)
joint_range_data=range(0,156)
input_image_size=(80,160)
time_window_size=20
latent_dim=64
dof=7
#batch_size=32


def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, input_image_size)

def roi_rect(args):
    c, s = args
    lt = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c
    rb = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c + tf.tile(tf.expand_dims(s, 1), (1,2))
    roi = tf.concat([lt, rb], axis=1)
    roi3 = tf.expand_dims(roi, 0)
    return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

def roi_rect1(args):
    c, s = args
    lt = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c
    rb = tf.tile(tf.expand_dims(1-s, 1), (1,2)) * c + tf.tile(tf.expand_dims(s, 1), (1,2))
    return tf.concat([lt, rb], axis=1)

def embed(args):
    whole_images, roi_images, rois = args
    roi = rois[0][0]
    y1 = roi[0]
    x1 = roi[1]
    y2 = roi[2]
    x2 = roi[3]
    y = tf.cast(input_image_size[0] * y1, tf.int32)
    x = tf.cast(input_image_size[1] * x1, tf.int32)
    h = tf.cast(input_image_size[0] * (y2 - y1), tf.int32)
    w = tf.cast(input_image_size[1] * (x2 - x1), tf.int32)
    resized_roi_images = tf.image.resize(roi_images, (h,w))
    padded_roi_images = tf.image.pad_to_bounding_box(resized_roi_images, y, x, input_image_size[0], input_image_size[1])

    d = 1.0 # dummy
    mask = (resized_roi_images + d) / (resized_roi_images + d)
    fg_mask = tf.image.pad_to_bounding_box(mask, y, x, input_image_size[0], input_image_size[1])
    bg_mask = 1.0 - fg_mask
    merged_img = fg_mask * padded_roi_images + bg_mask * whole_images
    return merged_img


class ROIConditionedAutoEncoderModel(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(ROIConditionedAutoEncoderModel, self).__init__(*args, **kargs)
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]

        roi_pos = tf.random.uniform([batch_size, 2])
        roi_scale = tf.random.uniform([batch_size], minval=0.4, maxval=0.8)
        roi_param = tf.concat([roi_pos, tf.expand_dims(roi_scale, 1)], 1)
        rect = roi_rect1([roi_pos, roi_scale])

        with tf.GradientTape() as tape:
            y_pred = self([x,roi_param], training=True) # Forward pass
            loss = self.compute_loss(y, y_pred, rect)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]

        roi_pos = tf.random.uniform([batch_size, 2])
        roi_scale = tf.random.uniform([batch_size], minval=0.4, maxval=0.8)
        roi_param = tf.concat([roi_pos, tf.expand_dims(roi_scale, 1)], 1)
        rect = roi_rect1([roi_pos, roi_scale])

        y_pred = self([x,roi_param], training=False) # Forward pass
        loss = self.compute_loss(y, y_pred, rect)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y, y_pred, rect):
        y_cropped = crop_and_resize((y, rect))
        loss = tf.reduce_mean(tf.square(y_cropped - y_pred))
        return loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


class PredictionModel(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(PredictionModel, self).__init__(*args, **kargs)
        tracker_names = ['image_loss', 'joint_loss', 'loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

        self.n_steps = 5

    def generate_roi(self, data):
        x, y = data
        batch_size = tf.shape(x[0])[0]
        roi_pos = tf.random.uniform([batch_size, 2])
        roi_scale = tf.random.uniform([batch_size], minval=0.4, maxval=0.8)
        return roi_pos, roi_scale

    def do_predict(self, x, y, roi_pos, roi_scale, training):
        x_image, x_joint = x
        image_loss = 0.0
        joint_loss = 0.0

        roi_param = tf.concat([roi_pos, tf.expand_dims(roi_scale, 1)], 1)

        y_images, y_joints = y
        y_images_tr = tf.transpose(y_images.astype('float32'), [1,0,2,3,4])
        y_joints_tr = tf.transpose(y_joints.astype('float32'), [1,0,2])

        for n in range(self.n_steps):
            pred_image, pred_joint = self((x_image, x_joint, roi_param), training=training) # Forward pass

            rect = roi_rect1([roi_pos, roi_scale])
            y_image_aug = crop_and_resize((y_images_tr[n], rect))
            image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
            joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

            if n < self.n_steps:
                x_image_tr = tf.transpose(x_image, [1,0,2,3,4])
                x_image_tr_list = tf.unstack(x_image_tr)
                x_image_tr_list[1:].append(pred_image)
                x_image_tr = tf.stack(x_image_tr_list)
                x_image = tf.transpose(x_image_tr, [1,0,2,3,4])
                x_joint_tr = tf.transpose(x_joint, [1,0,2])
                x_joint_tr_list = tf.unstack(x_joint_tr)
                x_joint_tr_list[1:].append(pred_joint)
                x_joint_tr = tf.stack(x_joint_tr_list)
                x_joint = tf.transpose(x_joint_tr, [1,0,2])

        image_loss /= self.n_steps
        joint_loss /= self.n_steps
        loss = image_loss + joint_loss
        return image_loss, joint_loss, loss
    
    def train_step(self, data):
        x, y = data
        roi_pos, roi_scale = self.generate_roi(data)

        with tf.GradientTape() as tape:
            image_loss, joint_loss, loss = self.do_predict(x, y, roi_pos, roi_scale, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['image_loss'].update_state(image_loss)
        self.train_trackers['joint_loss'].update_state(joint_loss)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        roi_pos, roi_scale = self.generate_roi(data)

        image_loss, joint_loss, loss = self.do_predict(x, y, roi_pos, roi_scale, training=False)
  
        self.test_trackers['image_loss'].update_state(image_loss)
        self.test_trackers['joint_loss'].update_state(joint_loss)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    # def compute_loss(self, y, y_pred, x, rect):
    #     x_image, x_joint = x
    #     y_image, y_joint = y

    #     y_cropped = crop_and_resize((y_image, rect))
                          
    #     image_loss = tf.reduce_mean(tf.square(y_cropped - y_pred[0]))
    #     joint_loss = tf.reduce_mean(tf.square(y_joint - y_pred[1]))

    #     loss = image_loss + self.alpha * joint_loss
    #     return image_loss, joint_loss, loss

    # def set_joint_weight(self, alpha):
    #     self.alpha = alpha
    
    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())

class ROIEstimationModel(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, predictor, *args, **kargs):
        super(ROIEstimationModel, self).__init__(*args, **kargs)
        self.predictor = predictor
        tracker_names = ['loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)
        self.n_steps = 5

    def generate_roi(self, data):
        x, y = data
        batch_size = tf.shape(x[0])[0]
        roi_pos = tf.tile([[0.5, 0.5]], (batch_size, 1))
        roi_scale = tf.repeat(0.8, batch_size)
        return roi_pos, roi_scale

    def do_predict(self, x, y, roi_pos, roi_scale, training):
        x_image, x_joint = x
        # image_loss = 0.0
        joint_loss = 0.0
        
        roi_param = tf.concat([roi_pos, tf.expand_dims(roi_scale, 1)], 1)

        y_images, y_joints = y
        # y_images_tr = tf.transpose(y_images.astype('float32'), [1,0,2,3,4])
        # y_joints_tr = tf.transpose(y_joints.astype('float32'), [1,0,2])
        y_images_tr = tf.transpose(y_images, [1,0,2,3,4])
        y_joints_tr = tf.transpose(y_joints, [1,0,2])

        # predict ROI once
        y_pred_roi = self((x_image, x_joint, roi_param), training=training)        

        # then predict N steps to compute loss
        for n in range(self.n_steps):
            pred_image, pred_joint = self.predictor((x_image, x_joint, y_pred_roi), training=False)
            
            #rect = roi_rect1([roi_pos, roi_scale])
            #y_image_aug = crop_and_resize((y_images_tr[n], rect))
            #image_loss += tf.reduce_mean(tf.square(y_image_aug - pred_image))
            joint_loss += tf.reduce_mean(tf.square(y_joints_tr[n] - pred_joint))

            if n < self.n_steps:
                x_image_tr = tf.transpose(x_image, [1,0,2,3,4])
                x_image_tr_list = tf.unstack(x_image_tr)
                x_image_tr_list[1:].append(pred_image)
                x_image_tr = tf.stack(x_image_tr_list)
                x_image = tf.transpose(x_image_tr, [1,0,2,3,4])
                x_joint_tr = tf.transpose(x_joint, [1,0,2])
                x_joint_tr_list = tf.unstack(x_joint_tr)
                x_joint_tr_list[1:].append(pred_joint)
                x_joint_tr = tf.stack(x_joint_tr_list)
                x_joint = tf.transpose(x_joint_tr, [1,0,2])

        # image_loss /= self.n_steps
        joint_loss /= self.n_steps
        loss = joint_loss
        return loss
    
    def train_step(self, data):
        x, y = data
        roi_pos, roi_scale = self.generate_roi(data)

        with tf.GradientTape() as tape:
            loss = self.do_predict(x, y, roi_pos, roi_scale, training=True)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        roi_pos, roi_scale = self.generate_roi(data)

        loss = self.do_predict(x, y, roi_pos, roi_scale, training=False)

        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())

class CustomAugmentation(tf.keras.layers.Layer):
    def __init__(self, brightness_max_delta=0.2,
                     contrast_lower=0.8, contrast_upper=1.2,
                     hue_max_delta=0.05):
        super().__init__()
        self.brightness_max_delta = brightness_max_delta
        self.contrast_lower = contrast_lower
        self.contrast_upper = contrast_upper
        self.hue_max_delta = hue_max_delta

    def call(self, inputs, training=None):
        return K.in_train_phase(tf.map_fn(self.augment_per_image, inputs),
                                          inputs, training=training)

    def augment_per_image(self, img):
        img = tf.image.random_brightness(img, max_delta=self.brightness_max_delta)
        img = tf.image.random_contrast(img, lower=self.contrast_lower, upper=self.contrast_upper)
        img = tf.image.random_hue(img, max_delta=self.hue_max_delta)
        # random_shift only works for array and eager tensor
        # img = tf.keras.preprocessing.image.random_shift(img, 0.02, 0.02,
        #                                                     row_axis=0, col_axis=1, channel_axis=2)
        # tf.image.crop_to_bounding_box, tf.image.pad_to_bounding_box

        img = tfa.image.translate(img, translations=[80*0.05, 160*0.05], fill_mode='constant')
        return img

    def get_config(self):
        config = {
            "brightness_max_delta" : self.brightness_max_delta,
            "contrast_lower" : self.contrast_lower,
            "contrast_upper" : self.contrast_upper,
            "hue_max_delta" : self.hue_max_delta
            }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


def model_lstm(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=False, name='lstm'):
    roi_dim = 3
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof + roi_dim
    x = tf.keras.layers.concatenate([imgvec_input, joint_input])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(state_dim)(x)

    predicted_ivec = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(x)
    predicted_jvec = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:image_vec_dim+dof], output_shape=(dof,))(x)
    roi_param = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim+dof:], output_shape=(roi_dim,))(x)
    roi_param = tf.keras.layers.Dense(3, activation='sigmoid')(roi_param) # (x, y, s)

    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_param)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_param)
    roi = tf.keras.layers.Lambda(roi_rect)([center, scale])
    
    m = keras.Model([imgvec_input, joint_input], [predicted_ivec, predicted_jvec, roi], name=name)
    m.summary()
    return m

def model_lstm_no_split(time_window_size, image_vec_dim, dof, lstm_units=50, use_stacked_lstm=True, name='lstm'):
    imgvec_input = tf.keras.Input(shape=(time_window_size, image_vec_dim))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    state_dim = image_vec_dim + dof

    joint_input_noise = tf.keras.layers.GaussianNoise(0.05)(joint_input)
    
    x = tf.keras.layers.concatenate([imgvec_input, joint_input_noise])

    if use_stacked_lstm:
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)

    x = tf.keras.layers.LSTM(lstm_units)(x)
    x = tf.keras.layers.Dense(state_dim)(x)

    lstm = keras.Model([imgvec_input, joint_input], x, name=name)
    lstm.summary()
    return lstm

def model_roi_conditioned_autoencoder(input_image_shape, latent_dim, use_color_augmentation=False):
    image_input = tf.keras.Input(shape=(input_image_shape))
    roi_input = tf.keras.Input(shape=(3,))
    
    # convert an ROI param to a rectangular region
    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_input)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_input)
    rect = tf.keras.layers.Lambda(roi_rect1)([center, scale])

    # crop&resize
    roi_img = tf.keras.layers.Lambda(crop_and_resize, output_shape=input_image_shape) ([image_input, rect])
    
    if use_color_augmentation:
        roi_img = ColorAugmentation()(roi_img)

    encoded_img = model_encoder(input_image_shape, latent_dim)(roi_img)
    decoded_img = model_decoder(input_image_shape, latent_dim)(encoded_img)

    model = ROIConditionedAutoEncoderModel(inputs=[image_input, roi_input], outputs=[decoded_img], name='roi_cond_autoencoder')
    model.summary()

    return model

def model_prediction(input_image_shape, time_window_size, image_vec_dim, dof, use_color_augmentation=False):
    image_input = tf.keras.Input(shape=((time_window_size,) + input_image_shape))
    joint_input = tf.keras.Input(shape=(time_window_size, dof))
    roi_input = tf.keras.Input(shape=(3,))

    # convert an ROI param to a rectangular region
    center = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_input)
    scale = tf.keras.layers.Lambda(lambda x:x[:,2], output_shape=(1,))(roi_input)
    rect = tf.keras.layers.Lambda(roi_rect)([center, scale])

    # crop&resize
    roi_img = tf.keras.layers.TimeDistributed(tf.keras.layers.Lambda(crop_and_resize, output_shape=input_image_shape)) ([image_input, rect])

    if use_color_augmentation:
        roi_img = TimeDistributedColorAugmentation()(roi_img)

    # encode
    roi_encoding = model_time_distributed_encoder(input_image_shape, time_window_size, image_vec_dim)(roi_img)

    # LSTM
    predicted_state = model_lstm_no_split(time_window_size, image_vec_dim, dof)([roi_encoding, joint_input])

    # separete predicted 
    predicted_roi_encoding = tf.keras.layers.Lambda(lambda x:x[:,:image_vec_dim], output_shape=(image_vec_dim,))(predicted_state)
    predicted_joint_positions = tf.keras.layers.Lambda(lambda x:x[:,image_vec_dim:], output_shape=(dof,))(predicted_state)

    # decode
    predicted_roi_image = model_decoder(input_image_shape, image_vec_dim)(predicted_roi_encoding)

    predictor = PredictionModel(inputs=[image_input, joint_input, roi_input],
                                outputs=[predicted_roi_image, predicted_joint_positions],
                                name='predictor')
    predictor.summary()

    x = tf.keras.layers.Dense(32, activation='selu')(predicted_state)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(8, activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    estimated_roi_params = tf.keras.layers.Dense(3, activation='sigmoid')(x)
    roi_estimator = ROIEstimationModel(predictor,
                                       inputs=[image_input, joint_input, roi_input],
                                       outputs=[estimated_roi_params],
                                       name='roi_estimator')
    roi_estimator.summary()    

    return predictor, roi_estimator
    

model_ae = model_roi_conditioned_autoencoder(input_image_size+(3,), latent_dim)
predictor, roi_estimator = model_prediction(input_image_size+(3,), time_window_size, latent_dim, dof)


def train_ae():
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_ae, train_ds, val_ds)
    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)

def train_predictor(cp='ae_cp.reaching-real.roi_cond_autoencoder.20220526184017'):
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)

    # load AE weights and copy them to the encoder and decoder of AE-LSTM model
    d = os.path.join(os.path.dirname(os.getcwd()), 'runs')
    model_ae.load_weights(os.path.join(d, cp, 'cp.ckpt'))
    predictor.get_layer('encoder').set_weights(model_ae.get_layer('encoder').get_weights())
    predictor.get_layer('decoder').set_weights(model_ae.get_layer('decoder').get_weights())

    tr = trainer.TimeSequenceTrainer(predictor, train_ds, val_ds, time_window_size=time_window_size)
    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)

# def train(cp='', epochs=800, alpha=1.0):
#     train_ds = Dataset(dataset, joint_range_data=joint_range_data)
#     train_ds.load(groups=train_groups, image_size=input_image_size)
#     train_ds.preprocess(time_window_size)
#     val_ds = Dataset(dataset, joint_range_data=joint_range_data)
#     val_ds.load(groups=val_groups, image_size=input_image_size)
#     val_ds.preprocess(time_window_size)

#     if cp != '':
#         tr = trainer.Trainer(predictor, train_ds, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
#     else:
#         tr = trainer.Trainer(predictor, train_ds, val_ds, time_window_size=time_window_size)

#     tr.model.set_joint_weight(alpha)
#     tr.train(epochs=epochs, early_stop_patience=800, reduce_lr_patience=100)
#     return tr

def train_roi_estimator(predictor_cp='ae_cp.reaching-real.predictor.20220527174859'):
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    train_ds.preprocess(time_window_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)

    roi_estimator.get_layer('encoder').trainable = False
    roi_estimator.get_layer('lstm').trainable = False

    tr = trainer.TimeSequenceTrainer(roi_estimator, train_ds, val_ds, time_window_size=time_window_size)
    d = os.path.join(os.path.dirname(os.getcwd()), 'runs')
    checkpoint_path = os.path.join(d, predictor_cp, 'cp.ckpt')
    print('load weights from ', checkpoint_path)
    predictor.load_weights(checkpoint_path)

    tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)


from mpl_toolkits.mplot3d import axes3d, Axes3D

class Predictor(trainer.TimeSequenceTrainer):
    def __init__(self, *args, time_window_size=20,**kargs):
        super(Predictor, self).__init__(*args, **kargs)

    def get_data(self):
        return next(self.val_gen)

    def get_a_sample_from_batch(self, batch, n):
        x,y = batch
        x_img = x[0][n:n+1]
        x_joint = x[1][n:n+1]
        y_img = y[0][n:n+1]
        y_joint = y[1][n:n+1]
        return (x_img,x_joint),(y_img,y_joint)

    ## evaluation of predictor
    def predict_with_roi(self, batch, roi_params):
        x,y = batch
        batch_sz = x[1].shape[0]
        if roi_params.ndim == 1:
            roi_params = np.tile(roi_params, (batch_sz,1))
        predicted_images, predicted_joints = self.model.predict(x + (roi_params,))
        #visualize_ds(y[0], roi_params) # to draw ROI roi_params must be converted to rectangle

        if y[0].ndim == 5:
            bboxes = roi_rect1((roi_params[:,:2], roi_params[:,2]))
            imgs = draw_bounding_boxes(x[0][:,-1], bboxes)
            visualize_ds(imgs)
        else:
            visualize_ds(y[0])
        visualize_ds(predicted_images)
        plt.show()

    def generate_roi_images(self, sample, n = 5):
        xs = np.linspace(0.1, 0.9, n)
        ys = np.linspace(0.1, 0.9, n)
        ss = 0.7 * np.sin(np.pi*xs)
        out_images = []
        for roi_params in zip(xs,ys,ss):
            roi_params = np.expand_dims(np.array(roi_params), 0)
            predicted_images, _ = self.model.predict(sample[0] + (roi_params,))
            imgs = sample[0][0][0][-1:]
            bboxes = roi_rect1((roi_params[:,:2], roi_params[:,2]))
            img_with_bb = draw_bounding_boxes(imgs, bboxes)[0]
            img = np.concatenate([img_with_bb, predicted_images[0]], axis=1)
            out_images.append(img)
            # plt.imshow(img)
            # plt.show()
        create_anim_gif_from_images(out_images, 'generated_roi_images.gif')

    def prediction_error(self, sample, roi_pos=[0.5,0.5], roi_scale=0.8):
        x,y = sample
        batch_size = tf.shape(x[0])[0]
        roi_pos = tf.tile([roi_pos], (batch_size, 1))
        roi_scale = tf.repeat(roi_scale, batch_size)
        image_loss, joint_loss, loss = self.model.do_predict(x, y, roi_pos, roi_scale, training=False)
        return image_loss, joint_loss, loss
        
    def prediction_errors(self, sample, nx=10, ny=10, ns=5):
        x = np.linspace(0.0, 1.0, nx, dtype='float32')
        y = np.linspace(0.0, 1.0, ny, dtype='float32')
        s = np.linspace(0.6, 0.8, ns, dtype='float32')
        x,y,s = np.meshgrid(x, y, s)
        z = np.array([self.prediction_error(sample, [x,y], s)[1] for x,y,s in zip(x.flatten(), y.flatten(), s.flatten())]).reshape((nx,ny,ns))

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
        ax.ticklabel_format(style='sci', axis='z', scilimits=(0,0))
        ax.plot_surface(x[:,:,sopt_idx], y[:,:,sopt_idx], z[:,:,sopt_idx], cmap='plasma')

        plt.show()
        return x,y,z,imgs[0]

    # evaluation of ROI estimator
    def predict_images(self, batch=None):
        if batch == None:
            x,y = next(self.val_gen)
        else:
            x,y = batch
        batch_size = tf.shape(x[0])[0]

        roi_params = np.tile([0.5,0.5,0.8], (batch_size,1))
        roi_params = self.model.predict(x+(roi_params,))
        bboxes = roi_rect1((roi_params[:,:2], roi_params[:,2]))
        
        imgs = draw_bounding_boxes(x[0][:,-1], bboxes)
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
        roi_params0 = np.tile([0.5,0.5,0.8], (batch_sz,1))
        pred_img, pred_joint = self.model.predict(x+(roi_params0,))
        bboxes = roi_rect1((roi_params0[:,:2], roi_params0[:,2]))
        return pred_img, pred_joint, bboxes

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
    
def prepare_for_test_roi_conditioned_ae(cp='ae_cp.reaching-real.roi_cond_autoencoder.20220526184017'):
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_ae, None, val_ds, checkpoint_file=cp)
    return tr
    
def prepare_for_test_predictor(cp='ae_cp.reaching-real.predictor.20220527174859'):
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = Predictor(predictor, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr

def prepare_for_test_roi_estimator(cp='ae_cp.reaching-real.roi_estimator.20220528172732'):
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    val_ds.preprocess(time_window_size)
    tr = Predictor(roi_estimator, None, val_ds, time_window_size=time_window_size, checkpoint_file=cp)
    return tr


# if __name__ == "__main__":
#     train_roi_estimator()
