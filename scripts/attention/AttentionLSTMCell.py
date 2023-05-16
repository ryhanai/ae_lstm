# -*- coding: utf-8 -*-

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import RNN
# from typeguard import typecheckd

from tensorflow_addons.utils.types import (
    Activation,
    FloatTensorLike,
    TensorLike,
    Initializer,
    Constraint,
    Regularizer,
)


# class AttentionLSTMCell(keras.layers.Layer):
class AttentionLSTMCell(keras.layers.LSTMCell):

    def __init__(
        self,
        units: TensorLike,
        activation: Activation = "tanh",
        recurrent_activation: Activation = "sigmoid",
        use_bias: bool = True,
        kernel_initializer: Initializer = "glorot_uniform",
        recurrent_initializer: Initializer = "orthogonal",
        bias_initializer: Initializer = "zeros",
        unit_forget_bias: bool = True,
        kernel_regularizer: Regularizer = None,
        recurrent_regularizer: Regularizer = None,
        bias_regularizer: Regularizer = None,
        kernel_constraint: Constraint = None,
        recurrent_constraint: Constraint = None,
        bias_constraint: Constraint = None,
        dropout: FloatTensorLike = 0.0,
        recurrent_dropout: FloatTensorLike = 0.0,
        norm_gamma_initializer: Initializer = "ones",
        norm_beta_initializer: Initializer = "zeros",
        norm_epsilon: FloatTensorLike = 1e-3,
        **kwargs,
    ):
        """Initializes the LSTM cell.
        Args:
          units: Positive integer, dimensionality of the output space.
          activation: Activation function to use. Default: hyperbolic tangent
            (`tanh`). If you pass `None`, no activation is applied (ie.
            "linear" activation: `a(x) = x`).
          recurrent_activation: Activation function to use for the recurrent
            step. Default: sigmoid (`sigmoid`). If you pass `None`, no
            activation is applied (ie. "linear" activation: `a(x) = x`).
          use_bias: Boolean, whether the layer uses a bias vector.
          kernel_initializer: Initializer for the `kernel` weights matrix, used
            for the linear transformation of the inputs.
          recurrent_initializer: Initializer for the `recurrent_kernel` weights
            matrix, used for the linear transformation of the recurrent state.
          bias_initializer: Initializer for the bias vector.
          unit_forget_bias: Boolean. If True, add 1 to the bias of the forget
            gate at initialization. Setting it to true will also force
            `bias_initializer="zeros"`. This is recommended in [Jozefowicz et
              al.](http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
          kernel_regularizer: Regularizer function applied to the `kernel`
            weights matrix.
          recurrent_regularizer: Regularizer function applied to
            the `recurrent_kernel` weights matrix.
          bias_regularizer: Regularizer function applied to the bias vector.
          kernel_constraint: Constraint function applied to the `kernel`
            weights matrix.
          recurrent_constraint: Constraint function applied to the
            `recurrent_kernel` weights matrix.
          bias_constraint: Constraint function applied to the bias vector.
          dropout: Float between 0 and 1. Fraction of the units to drop for the
            linear transformation of the inputs.
          recurrent_dropout: Float between 0 and 1. Fraction of the units to
            drop for the linear transformation of the recurrent state.
          norm_gamma_initializer: Initializer for the layer normalization gain
            initial value.
          norm_beta_initializer: Initializer for the layer normalization shift
            initial value.
          norm_epsilon: Float, the epsilon value for normalization layers.
          **kwargs: Dict, the other keyword arguments for layer creation.
        """
        super().__init__(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            **kwargs,
        )

        self.norm_gamma_initializer = keras.initializers.get(norm_gamma_initializer)
        self.norm_beta_initializer = keras.initializers.get(norm_beta_initializer)
        self.norm_epsilon = norm_epsilon
        
        # self.kernel_norm = self._create_norm_layer("kernel_norm")
        # self.recurrent_norm = self._create_norm_layer("recurrent_norm")
        # self.state_norm = self._create_norm_layer("state_norm")

    def build(self, input_shape):
        in_img_shape = input_shape[0]
        in_jv_shape = input_shape[1]
        super().build([None, self.units])

        def maybe_build_sublayer(sublayer, build_shape):
            if not sublayer.built:
                with tf.keras.backend.name_scope(sublayer.name):
                    sublayer.build(build_shape)
                    sublayer.built = True

        dof = in_jv_shape[1]
        h = in_img_shape[1]
        w = in_img_shape[2]

        channel = in_img_shape[3]
        self._h_dim = 32
        self.ratio = 8
        self.dense_h0 = keras.layers.Dense(self._h_dim, name="dense_h0", activation='sigmoid')
        self.mlp_0 = keras.layers.Dense((channel + self._h_dim)/self.ratio, name="mlp_0", activation='relu')
        self.mlp_1 = keras.layers.Dense(channel, name="mlp_1")

        sp_attention_kernel_size = 7  # 7 in CBAM
        self._n_filters = [32, 64]
        self.dense_h1 = keras.layers.Dense(h*w, name="dense_h1", activation='sigmoid')
        self.sp_conv = keras.layers.Conv2D(1, kernel_size=sp_attention_kernel_size, strides=[1, 1], padding="same", name="sp_conv")
        self.attention_conv0 = keras.layers.Conv2D(self._n_filters[0], kernel_size=3, strides=[2, 2], padding="same", activation="relu", name="attention_conv0")
        self.attention_bn0 = keras.layers.BatchNormalization(name="attention_bn0")
        self.attention_conv1 = keras.layers.Conv2D(self._n_filters[1], kernel_size=3, strides=[2, 2], padding="same", activation="relu", name="attention_conv1")
        self.attention_bn1 = keras.layers.BatchNormalization(name="attention_bn1")
        self.dense_lv = keras.layers.Dense(self.units - dof, name="dense_lv", activation='relu')

        maybe_build_sublayer(self.dense_h1, [self.units,])
        maybe_build_sublayer(self.sp_conv, [h, w, 3])
        maybe_build_sublayer(self.attention_conv0, in_img_shape)
        maybe_build_sublayer(self.attention_bn0, [None, int(h/2), int(w/2), self._n_filters[0]])
        maybe_build_sublayer(self.attention_conv1, [None, int(h/2), int(w/2), self._n_filters[0]])
        maybe_build_sublayer(self.attention_bn0, [None, int(h/4), int(w/4), self._n_filters[1]])
        maybe_build_sublayer(self.dense_lv, [int(h/4) * int(w/4) * self._n_filters[1],])

    def channel_attention(self, in_img, h_in):
        h1 = self.dense_h0(h_in)
        h1 = h1[:, tf.newaxis, tf.newaxis, :]

        channel = in_img.get_shape()[-1]
        avg_pool = tf.reduce_mean(in_img, axis=[1, 2], keepdims=True)
        avg_pool = tf.concat([avg_pool, h1], axis=3)
        assert avg_pool.get_shape()[1:] == (1, 1, channel+self._h_dim)
        avg_pool = self.mlp_0(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, (channel+self._h_dim)//self.ratio)
        avg_pool = self.mlp_1(avg_pool)
        assert avg_pool.get_shape()[1:] == (1, 1, channel)

        max_pool = tf.reduce_mean(in_img, axis=[1, 2], keepdims=True)
        max_pool = tf.concat([max_pool, h1], axis=3)
        assert max_pool.get_shape()[1:] == (1, 1, channel+self._h_dim)
        max_pool = self.mlp_0(max_pool)
        assert max_pool.get_shape()[1:] == (1, 1, (channel+self._h_dim)//self.ratio)
        max_pool = self.mlp_1(max_pool)
        assert max_pool.get_shape()[1:] == (1, 1, channel)

        scale = tf.sigmoid(avg_pool + max_pool, 'sigmoid')
        return scale

    def spatial_attention(self, in_img, h_in):
        h1 = self.dense_h1(h_in)
        # h1 = tf.reshape(h1, shape=[-1, in_img.get_shape()[1], in_img.get_shape()[2], 1])
        h1 = tf.reshape(h1, shape=[-1, tf.shape(in_img)[1], tf.shape(in_img)[2], 1])
        avg_pool = tf.reduce_mean(in_img, axis=[3], keepdims=True)
        assert avg_pool.get_shape()[-1] == 1
        max_pool = tf.reduce_max(in_img, axis=[3], keepdims=True)
        assert max_pool.get_shape()[-1] == 1
        h_input = tf.concat([avg_pool, max_pool, h1], axis=3)
        assert h_input.get_shape()[-1] == 3

        w0 = self.sp_conv(h_input)
        assert w0.get_shape()[-1] == 1

        scale = tf.sigmoid(w0, 'sigmoid')
        return scale

    def call(self, inputs, states, training=None):
        h_in = states[0]  # previous memory state
        c_in = states[1]  # previous carry state
        in_img = inputs[0]
        in_jv = inputs[1]

        ch_at = self.channel_attention(in_img, h_in)
        ch_at = ch_at * 0.8 + 0.2
        x = in_img * ch_at
        sp_at = self.spatial_attention(x, h_in)
        sp_at = sp_at * 0.8 + 0.2
        x = x * sp_at

        x = self.attention_conv0(x)
        x = self.attention_bn0(x)
        x = self.attention_conv1(x)
        x = self.attention_bn1(x)
        # shape = in_img.get_shape()
        shape = tf.shape(in_img)
        x = tf.reshape(x, shape=[-1, int(shape[1] * shape[2] / 16 * self._n_filters[-1])])  # flatten
        x = self.dense_lv(x)

        input_vec = tf.concat([x, in_jv], axis=1)

        z = keras.backend.dot(input_vec, self.kernel)
        z += keras.backend.dot(h_in, self.recurrent_kernel)

        if self.use_bias:
            z = keras.backend.bias_add(z, self.bias)

        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_in)
        h = o * self.activation(c)
        return [h, sp_at], [h, c]

    def get_config(self):
        config = {
            "norm_gamma_initializer": keras.initializers.serialize(
                self.norm_gamma_initializer
            ),
            "norm_beta_initializer": keras.initializers.serialize(
                self.norm_beta_initializer
            ),
            "norm_epsilon": self.norm_epsilon,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def _create_norm_layer(self, name):
        return keras.layers.LayerNormalization(
            beta_initializer=self.norm_beta_initializer,
            gamma_initializer=self.norm_gamma_initializer,
            epsilon=self.norm_epsilon,
            name=name,
        )
