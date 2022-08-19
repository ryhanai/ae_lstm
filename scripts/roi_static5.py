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
import model
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
    lt = (1-s) * c
    rb = (1-s) * c + s
    return tf.concat([lt, rb], axis=1)

def roi_rect1_with_fixed_aspect_ratio(args):
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

class StaticAttentionEstimatorModel(tf.keras.Model):
    def __init__(self, *args, **kargs):
        super(StaticAttentionEstimatorModel, self).__init__(*args, **kargs)
        tracker_names = ['reconst_loss', 'pred_loss', 'area_loss', 'shape_loss', 'loss']
        self.train_trackers = {}
        self.test_trackers = {}
        for n in tracker_names:
            self.train_trackers[n] = keras.metrics.Mean(name=n)
            self.test_trackers[n] = keras.metrics.Mean(name='val_'+n)

    def train_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        input_noise = tf.random.uniform(shape=(batch_size, 2), minval=-1, maxval=1)

        with tf.GradientTape() as tape:
            y_pred = self((x, input_noise), training=True) # Forward pass
            rloss, ploss, aloss, sloss, loss = self.compute_loss(y, y_pred, input_noise)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.train_trackers['reconst_loss'].update_state(rloss)
        self.train_trackers['pred_loss'].update_state(ploss)
        self.train_trackers['area_loss'].update_state(aloss)
        self.train_trackers['shape_loss'].update_state(sloss)
        self.train_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.train_trackers.items()])

    def test_step(self, data):
        x, y = data
        batch_size = tf.shape(x)[0]
        input_noise = tf.zeros(shape=(batch_size, 2))

        y_pred = self((x, input_noise), training=False) # Forward pass
        rloss, ploss, aloss, sloss, loss = self.compute_loss(y, y_pred, input_noise)

        self.test_trackers['reconst_loss'].update_state(rloss)
        self.test_trackers['pred_loss'].update_state(ploss)
        self.test_trackers['area_loss'].update_state(aloss)
        self.test_trackers['shape_loss'].update_state(sloss)
        self.test_trackers['loss'].update_state(loss)
        return dict([(trckr[0], trckr[1].result()) for trckr in self.test_trackers.items()])

    def compute_loss(self, y, y_pred, input_noise):
        y_aug = model.translate_image(y, input_noise)
        y_pred_img = y_pred[0]
        y_decoded_img = y_pred[1]
        roi_params = y_pred[3]
        reconstruction_loss = tf.reduce_mean(tf.square(y_aug - y_decoded_img))
        prediction_loss = tf.reduce_mean(tf.square(y_aug - y_pred_img))
        area_loss = tf.reduce_mean(roi_params[:,2]*roi_params[:,3]) + tf.reduce_mean(roi_params[:,6]*roi_params[:,7])
        shape_loss = tf.reduce_mean(tf.abs(tf.math.log(roi_params[:,2]) - tf.math.log(2*roi_params[:,3]))
                                        + tf.abs(tf.math.log(roi_params[:,6]) - tf.math.log(2*roi_params[:,7])))
        #loss = reconstruction_loss + prediction_loss + area_loss * 1e-3 + shape_loss * 1e-2
        loss = reconstruction_loss * 0. + prediction_loss + area_loss * 0. + shape_loss * 1e-3
        return reconstruction_loss, prediction_loss, area_loss, shape_loss, loss

    @property
    def metrics(self):
        return list(self.train_trackers.values()) + list(self.test_trackers.values())


def conv_block(x, out_channels, with_pooling=True):
    x = tf.keras.layers.Conv2D(out_channels, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if with_pooling:
        x = tf.keras.layers.MaxPool2D(pool_size=2)(x)
    return x

def model_encoder(input_shape, name='encoder'):
    image_input = tf.keras.Input(shape=(input_shape))
    
    x = conv_block(image_input, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64, with_pooling=False)
    x = conv_block(x, 128, with_pooling=False)
    
    encoder = keras.Model([image_input], x, name=name)
    encoder.summary()
    return encoder

def model_encoder2(input_shape, name='encoder2'):
    image_input = tf.keras.Input(shape=(input_shape))
    
    x = conv_block(image_input, 8)
    x = conv_block(x, 16)
    x = conv_block(x, 32)
    x = conv_block(x, 64)
    
    encoder = keras.Model([image_input], x, name=name)
    encoder.summary()
    return encoder

def deconv_block(x, out_channels, with_upsampling=True):
    x = tf.keras.layers.Conv2DTranspose(out_channels, kernel_size=3, strides=1, padding='same', activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    if with_upsampling:
        x = tf.keras.layers.UpSampling2D()(x)
    return x

def model_decoder(input_shape, name='decoder'):
    input_feature = tf.keras.Input(shape=(input_shape))
    x = deconv_block(input_feature, 64, with_upsampling=False)
    x = deconv_block(x, 32)
    x = deconv_block(x, 16)
    decoded_img = tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='sigmoid')(x)
    
    decoder = keras.Model([input_feature], decoded_img, name=name)    
    decoder.summary()
    return decoder

def model_ae(input_image_shape, name='autoencoder'):
    image_input = tf.keras.Input(shape=(input_image_shape))

    encoded_img = model_encoder(input_image_shape)(image_input)
    decoded_img = model_decoder(input_shape=(20,40,128))(encoded_img)

    m = Model(inputs=[image_input], outputs=[decoded_img], name=name)
    m.summary()
    return m

def model_roi_estimator(input_image_shape, name='roi_estimator'):
    image_input = tf.keras.Input(shape=(input_image_shape))

    encoded_img = model_encoder(input_image_shape)(image_input)
    x = tf.keras.layers.Flatten()(encoded_img)
    x = tf.keras.layers.Dense(64, activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(16, activation='selu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    roi_params = tf.keras.layers.Dense(6, activation='sigmoid')(x)
    
    model = Model(inputs=[image_input], outputs=[roi_params], name=name)
    model.summary()
    return model

def mask_images(args, filter_shape=(5,5), sigma=4.0):
    img, r1, r2 = args

    def aux_fn(args):
        img, r = args
        ishape = tf.shape(img)
        ih = tf.cast(ishape[0], tf.float32)
        iw = tf.cast(ishape[1], tf.float32)
        y1 = r[0]
        x1 = r[1]
        y2 = r[2]
        x2 = r[3]
        y = tf.cast(ih * y1, tf.int32)
        x = tf.cast(iw * x1, tf.int32)
        h = tf.cast(ih * (y2 - y1), tf.int32)
        w = tf.cast(iw * (x2 - x1), tf.int32)
        eps = 5
        h = tf.maximum(h, eps)
        w = tf.maximum(w, eps)
        y = tf.minimum(y, ishape[0] - h)
        x = tf.minimum(x, ishape[1] - w)
        ones = tf.ones((h, w, ishape[2]))
        mask = tf.image.pad_to_bounding_box(ones, y, x, ishape[0], ishape[1])
        return mask

    mask1 = tf.map_fn(fn=aux_fn, elems=(img, r1), dtype=tf.float32)
    mask2 = tf.map_fn(fn=aux_fn, elems=(img, r2), dtype=tf.float32)
    mask = tf.math.maximum(mask1, mask2)

    mask = tfa.image.gaussian_filter2d(mask, filter_shape=filter_shape, sigma=sigma)

    return mask * img + 0.2 * img


def model_static_attention_estimator(input_image_shape, noise_stddev=0.2, use_color_augmentation=False, use_geometrical_augmentation=True, name='static_attention_estimator'):
    image_input = tf.keras.Input(shape=(input_image_shape))
    input_noise = tf.keras.Input(shape=(2,))
    
    x = image_input

    if use_color_augmentation:
        x = model.ColorAugmentation()(x)
    if use_geometrical_augmentation:
        x = model.GeometricalAugmentation()(x, input_noise)

    x = tf.keras.layers.GaussianNoise(noise_stddev)(x)

    encoded_img = model_encoder(input_image_shape, name='encoder')(x)
    encoded_img_shape=(20,40,128)
    decoded_img = model_decoder(encoded_img_shape, name='decoder')(encoded_img)

    # attention map version
    masked_img = tf.keras.layers.Conv2D(128, kernel_size=1, strides=1, padding='same', activation='sigmoid')(x)

    # parametric ROI version
    # x = conv_block(encoded_img, 128)
    # x = conv_block(x, 128)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dense(16, activation='selu')(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    # roi_params = tf.keras.layers.Dense(8, activation='sigmoid')(x)

    # c1 = tf.keras.layers.Lambda(lambda x:x[:,:2], output_shape=(2,))(roi_params)
    # s1 = tf.keras.layers.Lambda(lambda x:x[:,2:4], output_shape=(2,))(roi_params)
    # c2 = tf.keras.layers.Lambda(lambda x:x[:,4:6], output_shape=(2,))(roi_params)
    # s2 = tf.keras.layers.Lambda(lambda x:x[:,7:], output_shape=(2,))(roi_params)
    # r1 = tf.keras.layers.Lambda(roi_rect1)([c1, s1])
    # r2 = tf.keras.layers.Lambda(roi_rect1)([c2, s2])
    # masked_img = tf.keras.layers.Lambda(mask_images)([encoded_img, r1, r2])

    predicted_img = model_decoder(encoded_img_shape, name='predictor')(masked_img) # predict the masked part of the image
    
    #m = Model(inputs=[image_input, input_noise], outputs=[encoded_img1, encoded_img2, roi_params], name=name)
    m = StaticAttentionEstimatorModel(inputs=[image_input, input_noise],
                                          outputs=[predicted_img, decoded_img, masked_img, roi_params], name='static_attention_estimator')
    m = StaticAttentionEstimatorModel(inputs=[image_input, input_noise],
                                          outputs=[predicted_img, decoded_img, masked_img, roi_params], name='static_attention_estimator')
    m.summary()
    return m


model_sae = model_static_attention_estimator(input_image_size+(3,))

def train_sae():
    train_ds = Dataset(dataset, joint_range_data=joint_range_data)
    train_ds.load(groups=train_groups, image_size=input_image_size)
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_sae, train_ds, val_ds)
    # tr.train(epochs=800, early_stop_patience=800, reduce_lr_patience=100)
    tr.train_prediction_task(epochs=800, early_stop_patience=800, reduce_lr_patience=100)
    return tr

def prepare_for_test_sae(cp='ae_cp.reaching-real.static_attention_estimator.20220614180752'):
    # 20220614173114
    # 20220614154818
    val_ds = Dataset(dataset, joint_range_data=joint_range_data)
    val_ds.load(groups=val_groups, image_size=input_image_size)
    tr = trainer.Trainer(model_sae, None, val_ds, checkpoint_file=cp)
    return tr

def test(tr):
    xs = tr.val_imgs[np.random.randint(0,1000,20)]
    noise = tf.zeros((xs.shape[0],2))
    y_pred = tr.model.predict((xs,noise))
    cs = y_pred[3][:,:2]
    ss = y_pred[3][:,2:4]
    cs2 = y_pred[3][:,4:6]
    ss2 = y_pred[3][:,6:8]
    a = draw_bounding_boxes(y_pred[0], roi_rect1((cs, ss)))
    b = draw_bounding_boxes(a, roi_rect1((cs2, ss2)))
    visualize_ds(b)
    visualize_ds(xs)
    plt.show()
    return xs, b

#if __name__ == "__main__":
#    train_sae()

# val_ds = Dataset(dataset, joint_range_data=joint_range_data)
# val_ds.load(groups=val_groups, image_size=input_image_size)
# imgs = np.array(val_ds.data[0][1])
# roi_imgs1 = imgs[:4]
# roi_imgs2 = imgs[4:8]
# bg_imgs = imgs[8:12]

# r1 = np.array([[0.2,0.2,0.7,0.6],
#                [0.1,0.2,0.6,0.7],
#                [0.3,0.3,0.5,0.5],
#                [0.0,0.1,0.4,0.6]], dtype='float32')
# r2 = np.array([[0.5,0.5,0.9,0.9],
#                [0.4,0.5,0.8,0.8],
#                [0.2,0.4,0.5,0.5],
#                [0.6,0.5,0.9,0.8]], dtype='float32')

#mask_images((roi_img1,roi_img2,r1,r2,bg_img))
