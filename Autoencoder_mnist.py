"""
Created on Sat Jul 10 18:21:07 2021

@author: Manish
"""

from keras.datasets import mnist
import numpy as np
from PIL import Image as im
from keras.layers import Input, Dense
from keras.models import Model



(train_images, _), (test_images,_) = mnist.load_data()


data = im.fromarray(train_images[1])

#normalize dataset

train_images = train_images.astype('float32')/255
test_images = test_images.astype('float32')/255

#reshaPE the images to 28X28 to 1d - 784 
train_images = train_images.reshape((len(train_images),   
                                     np.prod(train_images.shape[1:])))


test_images = test_images.reshape((len(test_images),   
                                     np.prod(test_images.shape[1:])))


#autoencoder creation

encoding_dim = 32
input_layer = Input (shape = (784,))
encoder_layer = Dense(encoding_dim, activation='relu')(input_layer)

decoder_layer = Dense(784, activation="sigmoid")(encoder_layer)

autoencoder = Model(input_layer, decoder_layer)

autoencoder.summary()

#training the model
#create encoder decoder separately 

encoder = Model(input_layer, encoder_layer) # input will be produce compressed code

encoded_input = Input(shape = (encoding_dim,)) #for decoder create encoded input
decoder_layer = autoencoder.layers[-1]
decoder = Model(encoded_input, decoder_layer(encoded_input))

#lets compile the autoencoder. 

autoencoder.compile(optimizer = 'adadelta', loss = 'binary_crossentropy')

epochs = 60
batch_size = 256
#kind of unsupervised, more like self supervised
autoencoder.fit(train_images, train_images, 
                epochs = 60, 
                batch_size = 256,
                shuffle = True, 
                validation_data = (test_images, test_images))




# lets use encoder and decoder toencode and decode our images , 


encoded_img = encoder.predict(test_images)

decode_imgs = decoder.predict(encoded_img)

