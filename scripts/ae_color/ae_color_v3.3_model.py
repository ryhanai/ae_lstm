# https://www.tensorflow.org/tutorials/generative/autoencoder
# put the Activation layer AFTER the BatchNormalization() layer


class Autoencoder(tf.keras.Model):

  def __init__(self, latent_dim):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
  
    self.encoder = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(width, height, 3)),
          
          tf.keras.layers.Conv2D(8, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.MaxPool2D(padding='same'),
          
          tf.keras.layers.Conv2D(16, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.MaxPool2D(padding='same'),          

          tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.MaxPool2D(padding='same'),          
          
          tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(dropout),

          tf.keras.layers.Flatten(),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.Dropout(dropout),

          tf.keras.layers.Dense(1000, activation='tanh'),
          tf.keras.layers.BatchNormalization(),

          tf.keras.layers.Dense(self.latent_dim, activation='tanh'),
          tf.keras.layers.BatchNormalization(),

        ])
  
    self.decoder = tf.keras.Sequential([
          tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),

          tf.keras.layers.Dense(1000, activation='tanh'),
          tf.keras.layers.BatchNormalization(),

          tf.keras.layers.Dense(size_*size_*channels_, activation='tanh'),
          # tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dropout(dropout),

          tf.keras.layers.Reshape(target_shape=(size_, size_, channels_)),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.Dropout(dropout),

          tf.keras.layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.UpSampling2D(),

          tf.keras.layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.UpSampling2D(),
          
          tf.keras.layers.Conv2DTranspose(16, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.UpSampling2D(),
          
          tf.keras.layers.Conv2DTranspose(8, kernel_size=3, strides=2, padding='same', activation='relu'),
          tf.keras.layers.BatchNormalization(),
          # tf.keras.layers.UpSampling2D(),
          
          tf.keras.layers.Conv2DTranspose(3, kernel_size=3, strides=1, padding='same', activation='relu'), #sigmoid
          tf.keras.layers.BatchNormalization(),

        ])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    
    return decoded


# initialize the model
latent_dim = 100
size_ = 20
channels_ = 64
dropout = 0.25

gaussian_auto_encoder = Autoencoder(latent_dim)

# opt = tf.keras.optimizers.Adam(learning_rate=0.005)
opt = tf.keras.optimizers.Adamax(learning_rate=0.05)

gaussian_auto_encoder.compile(loss='mse', optimizer=opt)

# see model summary
gaussian_auto_encoder.encoder.summary()
gaussian_auto_encoder.decoder.summary()


# Model: "sequential_34"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# conv2d_68 (Conv2D)           (None, 160, 160, 8)       224       
# _________________________________________________________________
# batch_normalization_229 (Bat (None, 160, 160, 8)       32        
# _________________________________________________________________
# conv2d_69 (Conv2D)           (None, 80, 80, 16)        1168      
# _________________________________________________________________
# batch_normalization_230 (Bat (None, 80, 80, 16)        64        
# _________________________________________________________________
# conv2d_70 (Conv2D)           (None, 40, 40, 32)        4640      
# _________________________________________________________________
# batch_normalization_231 (Bat (None, 40, 40, 32)        128       
# _________________________________________________________________
# conv2d_71 (Conv2D)           (None, 20, 20, 64)        18496     
# _________________________________________________________________
# batch_normalization_232 (Bat (None, 20, 20, 64)        256       
# _________________________________________________________________
# dropout_40 (Dropout)         (None, 20, 20, 64)        0         
# _________________________________________________________________
# flatten_17 (Flatten)         (None, 25600)             0         
# _________________________________________________________________
# batch_normalization_233 (Bat (None, 25600)             102400    
# _________________________________________________________________
# dense_68 (Dense)             (None, 1000)              25601000  
# _________________________________________________________________
# batch_normalization_234 (Bat (None, 1000)              4000      
# _________________________________________________________________
# dense_69 (Dense)             (None, 100)               100100    
# _________________________________________________________________
# batch_normalization_235 (Bat (None, 100)               400       
# =================================================================
# Total params: 25,832,908
# Trainable params: 25,779,268
# Non-trainable params: 53,640
# _________________________________________________________________
# Model: "sequential_35"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_70 (Dense)             (None, 1000)              101000    
# _________________________________________________________________
# batch_normalization_236 (Bat (None, 1000)              4000      
# _________________________________________________________________
# dense_71 (Dense)             (None, 25600)             25625600  
# _________________________________________________________________
# dropout_41 (Dropout)         (None, 25600)             0         
# _________________________________________________________________
# reshape_17 (Reshape)         (None, 20, 20, 64)        0         
# _________________________________________________________________
# batch_normalization_237 (Bat (None, 20, 20, 64)        256       
# _________________________________________________________________
# conv2d_transpose_85 (Conv2DT (None, 40, 40, 64)        36928     
# _________________________________________________________________
# batch_normalization_238 (Bat (None, 40, 40, 64)        256       
# _________________________________________________________________
# conv2d_transpose_86 (Conv2DT (None, 80, 80, 32)        18464     
# _________________________________________________________________
# batch_normalization_239 (Bat (None, 80, 80, 32)        128       
# _________________________________________________________________
# conv2d_transpose_87 (Conv2DT (None, 160, 160, 16)      4624      
# _________________________________________________________________
# batch_normalization_240 (Bat (None, 160, 160, 16)      64        
# _________________________________________________________________
# conv2d_transpose_88 (Conv2DT (None, 320, 320, 8)       1160      
# _________________________________________________________________
# batch_normalization_241 (Bat (None, 320, 320, 8)       32        
# _________________________________________________________________
# conv2d_transpose_89 (Conv2DT (None, 320, 320, 3)       219       
# _________________________________________________________________
# batch_normalization_242 (Bat (None, 320, 320, 3)       12        
# =================================================================
# Total params: 25,792,743
# Trainable params: 25,790,369
# Non-trainable params: 2,374
