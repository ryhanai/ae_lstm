import itertools
import numpy as np

import tensorflow as tf
from tensorflow import keras


def tf_crop_and_resize_batches_of_arrays(image_input, boxes_input, crop_size=(10, 10)):
    """
    The dimension of boxes is [batch, maxnumperbatch, 4]  # some can be just a bogus-zero-fill
    The dimension of the output should be [batch, maxnumperbatch, crop_0, crop_1, 1] crops.
    """

    bboxes_per_batch = tf.shape(boxes_input)[1]
    batch_size = tf.shape(boxes_input)[0]  # should be the same as image_input.shape[0]

    # the goal is to create a [batch, maxnumperbatch] field of values,
    #  which are the same across batch and equal to the batch_id
    # and then to reshape it in the same way as we do reshape the boxes_input to just tell tf about
    #  each bboxes batch (and image).
    index_to_batch = tf.tile(tf.expand_dims(tf.range(batch_size), -1), (1, bboxes_per_batch))

    # now both get reshaped as tf wants it:
    boxes_processed = tf.reshape(boxes_input, (-1, 4))
    box_ind_processed = tf.reshape(index_to_batch, (-1,))

    print('INDEX:')
    print(box_ind_processed)
    global INDEX
    INDEX = box_ind_processed
    # the method wants boxes = [num_boxes, 4], box_ind = [num_boxes] to index into the batch
    # the method returns [num_boxes, crop_height, crop_width, depth]

    tf_produced_crops = tf.image.crop_and_resize(
        image_input,
        boxes_processed,
        box_ind_processed,
        crop_size,
        method='bilinear',
        extrapolation_value=0,
        name=None
    )
    new_shape = tf.concat([tf.stack([batch_size, bboxes_per_batch]), tf.shape(tf_produced_crops)[1:]], axis=0)
    crops_resized_to_original = tf.reshape(tf_produced_crops,
                                           new_shape)
    return crops_resized_to_original


def keras_crop_and_resize_batches_of_arrays(image_input, boxes_input, crop_size=(10, 10)):
    """
    A helper function for tf_crop_and_resize_batches_of_arrays,
     assuming, that the crop_size would be a constant and not a tensorflow operation.
    """

    def f_crop(packed):
        image, boxes = packed
        return tf_crop_and_resize_batches_of_arrays(image, boxes, crop_size)

    return tf.keras.layers.Lambda(f_crop)([image_input, boxes_input])


def test_crops():
    # the intended usage:

    crop_size = (10, 10)

    image_input = np.ones((2, 200, 200, 1))
    boxes_input = np.array([[[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]],
                               [[0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 0.5, 0.5], [0.0, 0.0, 1.0, 1.0]]], dtype=np.float32)
    # the dimension of boxes is [batch, maxnumperbatch, 4]  # some can be just a bogus-zero-fill

    # image_input_ph = tf.placeholder(tf.float32, [None, None, None, None])
    # boxes_input_ph = tf.placeholder(tf.float32, [None, None, 4])
    # crop_result = sess.run(tf_crop_and_resize_batches_of_arrays(image_input_ph, boxes_input_ph, crop_size=crop_size),
    #                        feed_dict={image_input_ph: image_input, boxes_input_ph: boxes_input})

    crop_result = tf_crop_and_resize_batches_of_arrays(image_input, boxes_input, crop_size=crop_size)

    assert np.all(crop_result == 1.0), "when cropping image full of ones, we should get all ones too!"
    assert crop_result.shape == (boxes_input.shape[0], boxes_input.shape[1],
                                 crop_size[0], crop_size[1], image_input.shape[-1])

    return crop_result
