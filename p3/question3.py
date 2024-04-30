import numpy as np
import matplotlib.pyplot as plt 
from tensorflow.keras.datasets import mnist
import keras
from keras import layers
from keras.models import Model
import tensorflow as tf
import pdb
from tensorflow.keras.models import load_model


### a. Carry out a 2-D Haar wavelet decomposition.


(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

image = train_images[9]

sigma_z = np.sqrt(500)

noise = np.random.normal(0, sigma_z, image.shape)

noisy_image = image + noise

noisy_image_clip = np.clip(noisy_image, 0, 255)

plt.figure(figsize=(10, 5))

plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(1,2,2)
plt.title("Noisy Image")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')



import pywt

imagedecomp = noisy_image_clip.astype(np.float32)


LL, (LH, HL, HH) = pywt.dwt2(imagedecomp, 'haar')
figure, axis = plt.subplots(2, 2, figsize=(10, 10))

axis[0,0].imshow(LL, cmap = 'gray')
axis[0,0].set_title('Approximation Coefficients')
axis[0,0].axis('off')


axis[0,1].imshow(LH, cmap = 'gray')
axis[0,1].set_title('Horizontal Coefficients')
axis[0,1].axis('off')

axis[1,0].imshow(HL, cmap = 'gray')
axis[1,0].set_title('Vertical Coefficients')
axis[1,0].axis('off')


axis[1,1].imshow(HH, cmap = 'gray')
axis[1,1].set_title('Diagonal Coefficients')
axis[1,1].axis('off')



### c. Next threshold the wavelet coefficients for each of the following 
#levels, $T_1 = \frac{1}{10} \sigma_z \sqrt{\log(1024)}$, $T_2 = \frac{1}{2} \sigma_z \sqrt{\log(1024)}$, 
#and reconstruct the signal.

def threshold(coefficients, threshold):
    return pywt.threshold(coefficients, threshold, 'soft')

Threshold_1 = (1/10) * sigma_z * np.sqrt(np.log(1024))
Threshold_2 = (1/2) * sigma_z * np.sqrt(np.log(1024))


LH_Threshold1 = threshold(LH, Threshold_1)
HL_Threshold1 = threshold(HL, Threshold_1)
HH_Threshold1 = threshold(HH, Threshold_1)


LH_Threshold2 = threshold(LH, Threshold_2)
HL_Threshold2 = threshold(HL, Threshold_2)
HH_Threshold2 = threshold(HH, Threshold_2)

Threshold_1_Reconstructed = pywt.idwt2((LL, (LH_Threshold1, HL_Threshold1, HH_Threshold1)), 'haar')
Threshold_2_Reconstructed = pywt.idwt2((LL, (LH_Threshold2, HL_Threshold2, HH_Threshold2)), 'haar')


plt.figure(figsize=(15, 5))

plt.subplot(1,3,1)
plt.title("Original Noisy Image")
plt.imshow(noisy_image_clip, cmap='gray')
plt.axis('off')

plt.subplot(1,3,2)
plt.title("Threshold 1 Reconstructed")
plt.imshow(Threshold_1_Reconstructed, cmap='gray')
plt.axis('off')

plt.subplot(1,3,3)
plt.title("Threshold 2 Reconstructed")
plt.imshow(Threshold_2_Reconstructed , cmap='gray')
plt.axis('off')

plt.show()





### f. Now train a convolutional autoencoder model based on the architecture 
#in Part (1), The model should take the noisy images $y$ as input (instead of 
#$x$ as was in Part 1) and output an approximation $\hat{x}$ of the clean image $x$ 
#(So, your target output would remain same, i.e. $x$ as was in Part (1).

###################################################################################################
###FIRST TEST 8 FILTER LAYERS
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
    noise_factor = np.sqrt(0.75)
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
    
'''
Code to Visualize Encoding (Not posted in the project problem.)
'''


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


def eightFilterTwoLayerTest():   
    #FIRST TEST 8 FILTER LAYERS 
    ####################################################################################################################################################
    
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
        x=noisy_train_data,
        y=train_data,
        epochs=25,
        batch_size=128,
        shuffle=True,
        validation_data=(noisy_test_data, test_data),
    )
    
    ###################################################################################################
    
    '''
    Display noisy and denoised data
    '''
    predictions = autoencoder.predict(noisy_test_data)
    display(test_data, predictions, 'clean_vs_denoised')
    
    '''
    Bottleneck layer output
    '''
    bottleneck_model = Model(inputs=input, outputs=bottleneck_layer_output)
    bottleneck_encoding = bottleneck_model.predict(test_data)
    noisy_bottleneck_encoding = bottleneck_model.predict(noisy_test_data)
    prediction_bottleneck_encoding = bottleneck_model.predict(predictions)
    
    print("Visualizing Clean Encodings:")
    visualize_encoding(bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Noisy Encodings:")
    visualize_encoding(noisy_bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Denoised/Predicted Encodings:")
    visualize_encoding(prediction_bottleneck_encoding, test_label, [7, 8], 5)






def sixteenFilterTwoLayerTest():   
    #SECOND TEST 16 FILTER LAYERS 
    ####################################################################################################################################################
    encoder_Layer1_size = 16
    encoder_Layer2_size = 16
    decoder_Layer1_size = 16
    decoder_Layer2_size = 16
    
    
    
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
        x=noisy_train_data,
        y=train_data,
        epochs=25,
        batch_size=128,
        shuffle=True,
        validation_data=(noisy_test_data, test_data),
    )
    
    ###################################################################################################
    
    '''
    Display noisy and denoised data
    '''
    predictions = autoencoder.predict(noisy_test_data)
    display(test_data, predictions, 'clean_vs_denoised')
    
    '''
    Bottleneck layer output
    '''
    bottleneck_model = Model(inputs=input, outputs=bottleneck_layer_output)
    bottleneck_encoding = bottleneck_model.predict(test_data)
    noisy_bottleneck_encoding = bottleneck_model.predict(noisy_test_data)
    prediction_bottleneck_encoding = bottleneck_model.predict(predictions)
    
    print("Visualizing Clean Encodings:")
    visualize_encoding(bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Noisy Encodings:")
    visualize_encoding(noisy_bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Denoised/Predicted Encodings:")
    visualize_encoding(prediction_bottleneck_encoding, test_label, [7, 8], 5)


def eightFilterThreeLayerTest(): 
    #THIRD TEST 3 LAYERS 
    ####################################################################################################################################################
    
    encoder_Layer1_size = 8
    encoder_Layer2_size = 8
    decoder_Layer1_size = 8
    decoder_Layer2_size = 8
    
    
    
    '''
    Build model
    '''
    input = layers.Input(shape=(32, 32, 1))
    
    # Encoder

    
    x1c = layers.Conv2D(encoder_Layer1_size, (encoder_Layer1_kernel, encoder_Layer1_kernel), activation="relu", padding="same")(input)
    x1d = layers.MaxPooling2D((2, 2), padding="same")(x1c)
    x2c = layers.Conv2D(encoder_Layer2_size, (encoder_Layer2_kernel,encoder_Layer2_kernel), activation="relu", padding="same")(x1d)
    x2d = layers.MaxPooling2D((2, 2), padding="same")(x2c)
    x3c = layers.Conv2D(encoder_Layer2_size, (encoder_Layer2_kernel, encoder_Layer2_kernel), activation="relu", padding="same")(x2d)
    x3d = layers.MaxPooling2D((2, 2), padding="same")(x3c)

    '''
    Fully connected layers
    '''
    xf = layers.Flatten()(x3d)
    
    xld1 = layers.Dense(128, activation='relu')(xf)
    
    bottleneck_layer_size = 32
    
    bottleneck_layer_output = layers.Dense(bottleneck_layer_size, activation='relu')(xld1)
    
    
    # Decoder
    '''
    Fully connected layers
    '''
    xlu1 = layers.Dense(128, activation='relu')(bottleneck_layer_output)
    
    xlu2 = layers.Dense(encoder_Layer2_size*(32//8)*(32//8), activation='relu')(bottleneck_layer_output)
    
    xu = layers.Reshape((32//8, 32//8, encoder_Layer2_size))(xlu2)
    
    '''
    First upsampling layer (upsampling by a factor of 2 done using transposed convoltion)
    '''
    x = layers.Conv2DTranspose(decoder_Layer1_size, (decoder_Layer1_kernel, decoder_Layer1_kernel), strides=2, activation="relu", padding="same")(xu)
    '''
    Second/Third upsampling layer (upsampling by a factor of 2 done using transposed convoltion)
    '''
    x = layers.Conv2DTranspose(decoder_Layer2_size, (decoder_Layer2_kernel, decoder_Layer2_kernel), strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(decoder_Layer2_size, (decoder_Layer2_kernel, decoder_Layer2_kernel), strides=2, activation="relu", padding="same")(x)

   
    '''
    Last layer in the decoder to combine all channel outputs from the previous layer
    '''
    x = layers.Conv2D(1, (3, 3), activation="sigmoid", padding="same")(x)
    
    
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
        x=noisy_train_data,
        y=train_data,
        epochs=25,
        batch_size=128,
        shuffle=True,
        validation_data=(noisy_test_data, test_data),
    )
    
    ###################################################################################################
    
    '''
    Display noisy and denoised data
    '''
    predictions = autoencoder.predict(noisy_test_data)
    display(test_data, predictions, 'clean_vs_denoised')
    
    '''
    Bottleneck layer output
    '''
    bottleneck_model = Model(inputs=input, outputs=bottleneck_layer_output)
    bottleneck_encoding = bottleneck_model.predict(test_data)
    noisy_bottleneck_encoding = bottleneck_model.predict(noisy_test_data)
    prediction_bottleneck_encoding = bottleneck_model.predict(predictions)
    
    print("Visualizing Clean Encodings:")
    visualize_encoding(bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Noisy Encodings:")
    visualize_encoding(noisy_bottleneck_encoding, test_label, [7, 8], 5)
    print("Visualizing Denoised/Predicted Encodings:")
    visualize_encoding(prediction_bottleneck_encoding, test_label, [7, 8], 5)
    
    

print("8 Filter Layers 2 Layers ")
eightFilterTwoLayerTest()


print("16 Filter Layers 2 Layers ")
sixteenFilterTwoLayerTest()


print("8 Filter Layers 3 Layers ")
eightFilterThreeLayerTest()
    
    
