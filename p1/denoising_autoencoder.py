import numpy as np
import matplotlib.pyplot as plt

from keras import layers
from keras.datasets import mnist
from keras.models import Model
import tensorflow as tf
import pdb

###################################################################################################
'''
Configurable model parameters:
Configure these parameters before running the code to train the model
and check it's performance.
'''

'''
Change the number of convolutional kernels in the encoder/decoder layers
'''
encoder_Layer1_size = 8
encoder_Layer2_size = 8

decoder_Layer1_size = 8
decoder_Layer2_size = 8


'''
Change the kernel size in each of the layers
'''
encoder_Layer1_kernel = 3
encoder_Layer2_kernel = 3

decoder_Layer1_kernel = 3
decoder_Layer2_kernel = 3
###################################################################################################


def preprocess(array):
    """Normalizes the supplied array and reshapes it."""
    array = array.astype("float32") / 255.0
    array = np.reshape(array, (len(array), 32, 32, 1))
    return array


def noise(array):
    """Adds random noise to each image in the supplied array."""
    noise_factor = 0.75
    noisy_array = array + noise_factor * np.random.normal(
        loc=0.0, scale=1.0, size=array.shape
    )

    return np.clip(noisy_array, 0.0, 1.0)


def display(array1, array2,name):
    """Displays ten random images from each array."""
    n = 10
    indices = np.random.randint(len(array1), size=n)
    images1 = array1[indices, :]
    images2 = array2[indices, :]

    plt.close('all')
    plt.figure(figsize=(20, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):
        
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(image1.reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(image2.reshape(32, 32))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    # plt.show()
    plt.savefig(name+'.png')
    plt.close('all')

###################################################################################################
'''
Prepare datasets
'''
# Since we only need images from the dataset to encode and decode, we
# won't use the labels.
(train_data, train_label), (test_data, test_label) = mnist.load_data()
train_data, test_data = train_data[..., np.newaxis], test_data[..., np.newaxis]

# Reshape images from size 28x28 to 32x32 so that we can add more downsampling layers
# in the encoder and upsampling layers in the decoder later if required
train_data = tf.image.resize(train_data, (32,32)).numpy()
test_data = tf.image.resize(test_data, (32,32)).numpy()

# Normalize and reshape the data
train_data = preprocess(train_data)
test_data = preprocess(test_data)

# Create a copy of the data with added noise
# noisy_train_data = noise(train_data)
# noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
# display(train_data, noisy_train_data,'noisy_vs_orig')
# display(train_data, train_data,'noisy_vs_orig')
###################################################################################################

'''
Build model
'''
input = layers.Input(shape=(32, 32, 1))

# Encoder
'''
First convolutional layer
'''
x1c = layers.Conv2D(encoder_Layer1_size, (encoder_Layer1_kernel, encoder_Layer1_kernel), activation="relu", padding="same")(input)
'''
First downsampling layer (downsampling by a factor of 2)
'''
x1d = layers.MaxPooling2D((2, 2), padding="same")(x1c)
'''
Second convolutional layer
'''
x2c = layers.Conv2D(encoder_Layer2_size, (encoder_Layer2_kernel, encoder_Layer2_kernel), activation="relu", padding="same")(x1d)
'''
Second downsampling layer (downsampling by a factor of 2)
'''
x2d = layers.MaxPooling2D((2, 2), padding="same")(x2c)

'''
Fully connected layers
'''
xf = layers.Flatten()(x2d)

xld1 = layers.Dense(128, activation='relu')(xf)

bottleneck_layer_output = layers.Dense(32, activation='relu')(xld1)


# Decoder
'''
Fully connected layers
'''
xlu1 = layers.Dense(128, activation='relu')(bottleneck_layer_output)

xlu2 = layers.Dense(encoder_Layer2_size*(32//4)*(32//4), activation='relu')(bottleneck_layer_output)

xu = layers.Reshape((32//4, 32//4, encoder_Layer2_size))(xlu2)

'''
First upsampling layer (upsampling by a factor of 2 done using transposed convoltion)
'''
x1u = layers.Conv2DTranspose(decoder_Layer1_size, (decoder_Layer1_kernel, decoder_Layer1_kernel), strides=2, activation="relu", padding="same")(xu)
'''
Second upsampling layer (upsampling by a factor of 2 done using transposed convoltion)
'''
x2u = layers.Conv2DTranspose(decoder_Layer2_size, (decoder_Layer2_kernel, decoder_Layer2_kernel), strides=2, activation="relu", padding="same")(x1u)
'''
Last layer in the decoder to combine all channel outputs from the previous layer
'''
x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x2u)

# Autoencoder
autoencoder = Model(input, x)

autoencoder.compile(optimizer="adam", loss="binary_crossentropy")
autoencoder.summary()

###################################################################################################

'''
Train the model on clean data.
x is the input, while y is the outut
'''
autoencoder.fit(
    x=train_data,
    y=train_data,
    epochs=100,
    batch_size=128,
    shuffle=True,
    validation_data=(test_data, test_data),
)

###################################################################################################

'''
Display noisy and denoised data
'''
predictions = autoencoder.predict(test_data)
display(test_data, predictions,'input_vs_output')

'''
Bottleneck layer output
'''
bottleneck = Model(input,bottleneck_layer_output)
bottlenecck_encoding = bottleneck.predict(test_data)

'''
The code stops here in debugging mode. Type exit to finish execution.
'''
pdb.set_trace()
