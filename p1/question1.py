# -*- coding: utf-8 -*-
"""
Created on Sun Apr 21 16:34:59 2024

@author: cchu7
"""

#As is well known, the redundancy of the Continuous Wavelet Transform (CWT) may be 
#reduced by way of a wavelet frame representation of a function/signal. 
#This is further improved by way of orthonormal wavelet bases. 
#The vanishing moment property of a wavelet yields a condensed/compressed representation.

#A non-linear filtering procedure (such as denoising) is typically constructed by means of a thresholding strategy on the wavelet 
#representation coefficients aiming to eliminate undesired additive white Gaussian noise.

#To conduct a comparative analysis of two non-linear filtering- denoising approaches, one based on multiscale data analysis and 
#the other using a Neural-network-based auto-encoder. You will be asked to implement a standard CNN auto-encoder of varying complexity
# (number of channels and layers) trained with images signals and obtain a Discrete Wavelet Transform of data.

#A systematic learning/representation of data by a Convolutional auto-encoder yields the so-called ”bottleneck” compressed representation. 

#The first part of the project consists of using Matlab and an attached python code for auto-encoder. You will subsequently be experimenting
# with denoising data (time-series and images) and be comparing the two approaches fundamentally rooted in noise coefficients contribution 
#(much less compressed) and in signal coefficients much more compressed when representing signal information.

#As a result, a thresholding technique, as used in the orthonormal representation of data, eliminates the ”Only Noise” contributions and preserves
# the "Signal + Noise.". Our goal is to compare both non-linear filtering techniques.


#import any libraries we need. 
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft, fftfreq





#Generate/simulate a Haar Wavelet function and plot it along with its Fourier Transform, 
#and do so at 4 consecutive resolutions.

def haar(time, resolution):
    if resolution < time < 2 * resolution: 
        return 1
    elif 2 * resolution < time < 3 * resolution: 
        return -1
    else:
        return 0
    
t = np.linspace(0, 5, 1000)

resolutions = [0.25, 0.5, 0.75, 1]


wavelets = []

for resolution in resolutions:
    coefficients = []
    for time in t: 
        coefficient = haar(time, resolution)
        coefficients.append(coefficient)
    wavelets.append(np.array(coefficients))


plt.figure(figsize = (20, 20))

for i in range(len(wavelets)):
    wavelet = wavelets[i]
    plt.subplot(2, 2, i + 1)
    plt.plot(t, wavelet)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    

plt.tight_layout()
plt.show()
    



freqdomain = fftfreq(len(t), d=(t[1] - t[0]))


fft_wavelets = []
for wavelet in wavelets:
    fft_result = np.fft.fft(wavelet)
    abs_fft_result = np.abs(fft_result)
    fft_wavelets.append(abs_fft_result)    
    
plt.figure(figsize = (20,20))

for i, fft_wavelet in enumerate(fft_wavelets):
    plt.subplot(2, 2, i + 1)
    plt.plot(freqdomain, fft_wavelet)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Amplitude")
    plt.xlim(-25, 25)
    plt.grid(True)

plt.tight_layout()
plt.show()


#Generate/simulate again 4 resolutions of a Daubechies-4 wavelet and plot them along
#with their Fourier Transforms.

import pywt

daubechies4w = pywt.Wavelet('db4')
resolutions = [1, 2, 3, 4]

plt.figure(figsize = (20, 20))

for i, resolution in enumerate(resolutions):
    #\phi(t) = \sqrt{2} \sum_{k=0}^{N-1} h_k \phi(2t-k) (Scaling Function)
    #\psi(t) = \sqrt{2} \sum_{k=0}^{N-1} g_k \psi(2t-k) (Wavelet Function)
    plt.subplot(2, 2, i + 1)
    phi, psi, x = daubechies4w.wavefun(level=resolution)
    plt.plot(x, psi)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
plt.tight_layout()
plt.show()



plt.figure(figsize = (20, 20))

for i, resolution in enumerate(resolutions):
    plt.subplot(2, 2, i + 1)
    phi, psi, x = daubechies4w.wavefun(level=resolution)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Amplitude")
    freqdomain = fftfreq(psi.size, d=(x[1] - x[0]))
    fft_psi = np.fft.fft(psi)
    plt.plot(freqdomain, np.abs(fft_psi))
    
  
plt.tight_layout()
plt.show()
    

meyer = pywt.Wavelet('dmey')
resolutions = [1, 2, 3, 4]

plt.figure(figsize = (20, 20))

for i, resolution in enumerate(resolutions):
    plt.subplot(2, 2, i + 1)
    phi, psi, x = meyer.wavefun(level=resolution)
    plt.plot(x, psi)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    
plt.tight_layout()
plt.show()



plt.figure(figsize = (20, 20))

for i, resolution in enumerate(resolutions):
    plt.subplot(2, 2, i + 1)
    phi, psi, x = meyer.wavefun(level=resolution)
    plt.title("Resolution " + str(resolutions[i]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel("Amplitude")
    freqdomain = fftfreq(psi.size, d=(x[1] - x[0]))
    fft_psi = np.fft.fft(psi)
    plt.plot(freqdomain, np.abs(fft_psi))
    
  
plt.tight_layout()
plt.show()




#Take a CNN architecture for an auto-encoder (i.e. both encoder and decoder)
#of varying complexity. (see towardsdatascience.com/autoencoders-and-the-denoising-feature-from-theory-to-practice). 
#Use attached python code for auto-encoder. You will need to have NumPy, Matplotlib and 
#TensorFlow packages installed to run this code. Run the code to train a convolutional 
#autoencoder over clean MNIST dataset (You don’t need to download the dataset, the code \
#will do this automatically), The encoder part of your autoencoder will take clean images 
#as input and output a compressed encoded vector. The decoder will take this encoding as 
#input and again aim to output the original clean image. Visualize the vector output 
#(which is the compressed encoding) of the bottleneck layer for 5 samples 
#(sampled from the test set) each corresponding to two digit classes of your choice 
#(say 7 and 8).
    

'''
denoisingautoencoder.py as posted by the project requirements
'''
    
import keras
from keras import layers
from keras.datasets import mnist
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.models import load_model
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
noisy_train_data = noise(train_data)
noisy_test_data = noise(test_data)

# Display the train data and a version of it with added noise
display(train_data, noisy_train_data,'noisy_vs_orig')
display(train_data, train_data,'noisy_vs_orig')
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

bottleneck_layer_size = 32

bottleneck_layer_output = layers.Dense(bottleneck_layer_size, activation='relu')(xld1)


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
noisy_bottleneck_encoding = bottleneck.predict(noisy_test_data)





'''
Code to Visualize Encoding (Not posted in the project problem.)
'''

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def visualize_encoding(encodings, labels, digits, num_samples):
    sample_indices = {}
    for digit in digits: 
        indices = np.flatnonzero(labels == digit)
        sampled_indices = indices[:num_samples]
        sample_indices[digit] = sampled_indices

    fig, axes = plt.subplots(nrows=len(digits), ncols=num_samples, figsize=(num_samples * 2, len(digits) * 2))
    for row, digit in enumerate(digits):
        for col in range(num_samples):
            if len(digits) > 1: 
                ax = axes[row][col]
            else:
                ax = axes[col]
            encoding = encodings[sample_indices[digit][col]]
            ax.imshow(encoding.reshape(4, 8), cmap='gray')
            ax.set_title(f"Digit {digit}")
            ax.axis('off')
    plt.show()


autoencoder = load_model('autoencoder_model.h5')

print("Visualizing Clean Encodings:")
visualize_encoding(bottlenecck_encoding, test_label, [7, 8], 5)
print("Visualizing Noisy Encodings:")
visualize_encoding(noisy_bottleneck_encoding, test_label, [7, 8], 5)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    