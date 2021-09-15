import numpy as np
import threading
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class Iterator(object):
    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        self.reset()
        while True:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

class ImageRNNIterator(Iterator):
    def __init__(self, dataset, data_generator,
                     batch_size=32, time_window_size=20, shuffle=False, seed=None):
        self.ds = dataset
        self.data_generator = data_generator
        self.window_size = time_window_size

        self.indices = []
        for g_num, group in enumerate(self.ds):
            jv_seq, images = group
            n_seq = jv_seq.shape[0] - (self.window_size + 1)
            self.indices.extend([(g_num, i) for i in range(n_seq)])

        self.batch_index = 0
        self.total_batches_seen = 0
        super().__init__(len(self.indices), batch_size, shuffle, seed)

    def next(self, batch_size=32, shuffle=False, seed=None):
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
            if current_batch_size < batch_size:
                # model.fit is interrupted, when the size of generated batch is smaller than batch_size.
                # WARNING:tensorflow:Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches 
                index_array, current_index, current_batch_size = next(self.index_generator)

        # print(index_array)
        ishape = self.ds[0][1][0].shape
        jv_dim = self.ds[0][0].shape[1]
        batch_x_imgs = np.empty((current_batch_size, self.window_size, ishape[0], ishape[1], ishape[2]))
        batch_x_jvs = np.empty((current_batch_size, self.window_size, jv_dim))
        batch_y_img = np.empty((current_batch_size, ishape[0], ishape[1], ishape[2]))
        batch_y_jv = np.empty((current_batch_size, jv_dim))

        for i, seq_idx in enumerate(index_array):
            group_num, seq_num = self.indices[seq_idx]
            batch_x_jvs[i] = self.ds[group_num][0][seq_num:seq_num+self.window_size]
            batch_y_jv[i] = self.ds[group_num][0][seq_num+self.window_size]
            for j in range(self.window_size):
                batch_x_imgs[i][j] = self.ds[group_num][1][seq_num+j]
            batch_y_img[i] = self.ds[group_num][1][seq_num+self.window_size]

        roi = np.array([0.48, 0.25, 0.92, 0.75]) # [y1, x1, y2, x2] in normalized coodinates
        batch_rois = np.tile(roi, [batch_size, self.window_size, 1])

        return (batch_x_imgs, batch_x_jvs, batch_rois), (batch_y_img, batch_y_jv)

    def number_of_data(self):
        return len(self.indices)
    
class DPLGenerator(ImageDataGenerator):
    def __init__(self, featurewise_center = False, samplewise_center = False,
                 featurewise_std_normalization = False, samplewise_std_normalization = False,
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0,
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0,
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False,
                 vertical_flip = False, rescale = None, preprocessing_function = None,
                 data_format = None, validation_split = 0.0, random_crop = None):

        super().__init__(featurewise_center, samplewise_center,
                         featurewise_std_normalization, samplewise_std_normalization,
                         zca_whitening, zca_epsilon, rotation_range, width_shift_range,
                         height_shift_range, brightness_range, shear_range, zoom_range,
                         channel_shift_range, fill_mode, cval, horizontal_flip,
                         vertical_flip, rescale, preprocessing_function,
                         data_format, validation_split)

        assert random_crop == None or len(random_crop) == 2
        self.random_crop_size = random_crop

    def flow(self, dataset, batch_size=32, time_window_size=20, shuffle=True, seed=None):
        return ImageRNNIterator(dataset, self, batch_size, time_window_size, shuffle, seed)
    
    def get_random_transform(self, img_shape, seed=None):
        tf = super().get_random_transform(img_shape, seed)
        if self.random_crop_size != None:
            height, width = img_shape[0], img_shape[1]
            dy, dx = self.random_crop_size
            if img_shape[0] < dy or img_shape[1] < dx:
                raise ValueError(f"Invalid random_crop_size : original = {img_shape}, crop_size = {self.random_crop_size}")

            x = np.random.randint(0, width - dx + 1)
            y = np.random.randint(0, height - dy + 1)
            tf['crop'] = [(x, y), (dx, dy)]
        return tf

    def apply_transform(self, img, transform_parameters):
        img = super().apply_transform(img, transform_parameters)
        try:
            (x,y),(dx,dy) = transform_parameters['crop']
            assert img.shape[2] == 3
            return img[y:(y+dy), x:(x+dx), :]
        except:
            return img
