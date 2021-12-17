# -*- coding: utf-8 -*-

import os, sys, glob, re, time, copy
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model

from utils import *
from generator import *

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

input_image_shape = (90, 160, 3)
input_image_size = input_image_shape[:2]
roi_shape = (40, 80, 3)
roi_size = roi_shape[:2]
dense_dim = 512
latent_dim = 32
time_window_size = 20
dof = 7
batch_size = 32


def crop_and_resize(args):
    images, bboxes= args
    box_indices = tf.range(tf.size(bboxes)/4, dtype=tf.dtypes.int32)
    return tf.image.crop_and_resize(images, bboxes, box_indices, roi_size)

class AutoencoderWithCrop(tf.keras.Model):
    """
    override loss computation
    """
    def __init__(self, *args, **kargs):
        super(AutoencoderWithCrop, self).__init__(*args, **kargs)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="val_loss")

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

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False) # Forward pass

        roi = y_pred[2]
        print(' ROI: ', roi)

        val_loss = self.compute_loss(y, y_pred, x)

        self.val_loss_tracker.update_state(val_loss)
        return {"loss": self.val_loss_tracker.result()}

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
        return [self.loss_tracker, self.val_loss_tracker]

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

# def roi_rect(args):
#     """
#     Fixed ROI (averaged over estimated ROIs)
#     """
#     c, s = args
#     lt = tf.tile(tf.constant([[0.1205094, 0.15675367]], dtype=tf.float32), (batch_size,1))
#     rb = tf.tile(tf.constant([[0.70754665, 0.74379086]], dtype=tf.float32), (batch_size,1))
#     roi = tf.concat([lt, rb], axis=1)
#     roi3 = tf.expand_dims(roi, 0)
#     return tf.transpose(tf.tile(roi3, tf.constant([time_window_size, 1, 1], tf.int32)), [1,0,2])

def roi_rect(args):
    """
    To invalidate ROI, use this function.
    This returns the whole image as ROI
    """
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
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
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
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
    # x = tf.keras.layers.LSTM(state_dim, return_sequences=True)(x)
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
        self.train_ds = Dataset()
        self.train_ds.load('reaching', groups=range(1,300), image_size=input_image_size)
        self.train_ds.extend_initial_and_final_states(self.time_window)
        train_generator = DPLGenerator()
        self.train_gen = train_generator.flow(self.train_ds.get(),
                                                  const_roi_fun,
                                                  batch_size=self.batch_size,
                                                  time_window_size=self.time_window,
                                                  add_roi=False)

    def load_val_data(self):
        self.val_ds = Dataset()
        self.val_ds.load('reaching', groups=range(300,350), image_size=input_image_size)
        self.val_ds.extend_initial_and_final_states(self.time_window-2)
        val_generator = DPLGenerator()
        self.val_gen = val_generator.flow(self.val_ds.get(),
                                              const_roi_fun,
                                              batch_size=self.batch_size,
                                              time_window_size=self.time_window,
                                              add_roi=False)

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
        d = self.val_ds.data
        seq_len = len(d[group_num][1])
        ishape = d[group_num][1][0].shape
        jv_dim = d[group_num][0].shape[1]
        batch_size = seq_len-self.time_window
        batch_x_imgs = np.empty((batch_size, self.time_window, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size, ishape[0], ishape[1], ishape[2]))
        batch_y_jv = np.empty((batch_size, jv_dim))
        for i, seq_num in enumerate(range(batch_size)):
            batch_x_jvs[i] = d[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[i] = d[group_num][0][seq_num+self.time_window]
            batch_x_imgs[i] = d[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[i] = d[group_num][1][seq_num+self.time_window]

        # return (batch_x_imgs, batch_x_jvs), (batch_y_img, batch_y_jv)

        ximgs = batch_x_imgs[start:start+length:2]
        xjvs = batch_x_jvs[start:start+length:2]
        predicted_images, predicted_joint_positions, estimated_rois = self.model.predict((ximgs, xjvs))

        yimgs = batch_y_img[start:start+length:2]
        create_anim_gif_for_group(yimgs, estimated_rois[:,0,:], group_num)

    def process_sequence(self, group_num=0):
        res = []
        d = self.val_ds.data
        seq_len = len(d[group_num][1])
        ishape = d[group_num][1][0].shape
        jv_dim = d[group_num][0].shape[1]
        batch_size = 32
        batch_x_imgs = np.empty((batch_size, self.time_window, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((batch_size, self.time_window, jv_dim))
        batch_y_img = np.empty((batch_size, ishape[0], ishape[1], ishape[2]))
        batch_y_jv = np.empty((batch_size, jv_dim))

        for seq_num in range(seq_len-self.time_window):
            print(seq_num)
            batch_x_jvs[:] = d[group_num][0][seq_num:seq_num+self.time_window]
            batch_y_jv[:] = d[group_num][0][seq_num+self.time_window]
            batch_x_imgs[:] = d[group_num][1][seq_num:seq_num+self.time_window]
            batch_y_img[:] = d[group_num][1][seq_num+self.time_window]

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

