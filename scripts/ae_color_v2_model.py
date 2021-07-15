## ENCODER ##
encoder_input = keras.Input(shape=(width, height, 3)) 

e_conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(encoder_input)
pool1 = layers.MaxPooling2D((2, 2), padding='same')(e_conv1)
batchnorm_1 = layers.BatchNormalization()(pool1)

e_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(batchnorm_1)
pool2 = layers.MaxPooling2D((2, 2), padding='same')(e_conv2)
batchnorm_2 = layers.BatchNormalization()(pool2)

e_conv3 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(batchnorm_2)
encoder_output = layers.MaxPooling2D((2, 2), padding='same')(e_conv3)

encoder = keras.Model(encoder_input, encoder_output, name='encoder')
encoder.summary()


# initialize the model
size_=38
channels_=16

## DECODER ##
decoder_input = keras.Input(shape=(size_, size_, channels_), name='encoded_img')
reshape = layers.Reshape((size_, size_, channels_))(decoder_input)

d_conv1 = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(reshape)
up1 = layers.UpSampling2D((2, 2))(d_conv1)

d_conv2 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(up1)
up2 = layers.UpSampling2D((2, 2))(d_conv2)

d_conv3 = layers.Conv2D(64, (3, 3), activation='relu')(up2)
up3 = layers.UpSampling2D((2, 2))(d_conv3)

decoder_output = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(up3)

decoder = keras.Model(decoder_input, decoder_output, name='decoder')
decoder.summary()


autoencoder_input = keras.Input(shape=(width,height,3), name='img')
encoded_img = encoder(autoencoder_input)
decoded_img = decoder(encoded_img)

autoencoder = keras.Model(autoencoder_input, decoded_img, name='autoencoder')

opt = tf.keras.optimizers.Adam(learning_rate=1e-3) #Adamax
autoencoder.compile(loss='mse', metrics=["mae", "acc"], optimizer=opt)

autoencoder.summary()

# Model: "encoder"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# input_1 (InputLayer)         [(None, 300, 300, 3)]     0         
# _________________________________________________________________
# conv2d (Conv2D)              (None, 300, 300, 64)      1792      
# _________________________________________________________________
# max_pooling2d (MaxPooling2D) (None, 150, 150, 64)      0         
# _________________________________________________________________
# batch_normalization (BatchNo (None, 150, 150, 64)      256       
# _________________________________________________________________
# conv2d_1 (Conv2D)            (None, 150, 150, 32)      18464     
# _________________________________________________________________
# max_pooling2d_1 (MaxPooling2 (None, 75, 75, 32)        0         
# _________________________________________________________________
# batch_normalization_1 (Batch (None, 75, 75, 32)        128       
# _________________________________________________________________
# conv2d_2 (Conv2D)            (None, 75, 75, 16)        4624      
# _________________________________________________________________
# max_pooling2d_2 (MaxPooling2 (None, 38, 38, 16)        0         
# =================================================================
# Total params: 25,264
# Trainable params: 25,072
# Non-trainable params: 192
# _________________________________________________________________
# Model: "decoder"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# encoded_img (InputLayer)     [(None, 38, 38, 16)]      0         
# _________________________________________________________________
# reshape (Reshape)            (None, 38, 38, 16)        0         
# _________________________________________________________________
# conv2d_3 (Conv2D)            (None, 38, 38, 16)        2320      
# _________________________________________________________________
# up_sampling2d (UpSampling2D) (None, 76, 76, 16)        0         
# _________________________________________________________________
# conv2d_4 (Conv2D)            (None, 76, 76, 32)        4640      
# _________________________________________________________________
# up_sampling2d_1 (UpSampling2 (None, 152, 152, 32)      0         
# _________________________________________________________________
# conv2d_5 (Conv2D)            (None, 150, 150, 64)      18496     
# _________________________________________________________________
# up_sampling2d_2 (UpSampling2 (None, 300, 300, 64)      0         
# _________________________________________________________________
# conv2d_6 (Conv2D)            (None, 300, 300, 3)       1731      
# =================================================================
# Total params: 27,187
# Trainable params: 27,187
# Non-trainable params: 0
# _________________________________________________________________
# Model: "autoencoder"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# img (InputLayer)             [(None, 300, 300, 3)]     0         
# _________________________________________________________________
# encoder (Functional)         (None, 38, 38, 16)        25264     
# _________________________________________________________________
# decoder (Functional)         (None, 300, 300, 3)       27187     
# =================================================================
# Total params: 52,451
# Trainable params: 52,259
# Non-trainable params: 192
# _________________________________________________________________
