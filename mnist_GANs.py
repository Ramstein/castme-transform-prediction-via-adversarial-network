# keras adversarial GANs for forging mnist

import matplotlib as mpl
mpl.use('AGG')  # this  line allows matplotlib(mpl) to run with no display defined
import numpy as np
import pandas as pd, os


'''image_utils.py'''
import keras.backend as K
import numpy as np
from keras.layers import Input, Reshape

def dim_ordering_fix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 2, 3, 1))
def dim_ordering_unfix(x):
    if K.image_dim_ordering() == 'th':
        return x
    else:
        return np.transpose(x, (0, 3, 1, 2))
def dim_ordering_shape(input_shape):
    if K.image_dim_ordering() == 'th':
        return input_shape
    else:
        return (input_shape[1], input_shape[2], input_shape[0])
def dim_ordering_input(input_shape, name):
    if K.image_dim_ordering() == 'th':
        return Input(input_shape, name=name)
    else:
        return Input((input_shape[1], input_shape[2], input_shape[0]), name=name)
def dim_ordering_reshape(k, w, **kwargs):
    if K.image_dim_ordering() == 'th':
        return Reshape((k, w, w), **kwargs)
    else:
        return Reshape((w, w, k), **kwargs)
def channel_axis():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return 3


'''importing all of the modules'''

import keras.backend as k
from keras.layers import Dense, Reshape, Flatten, Dropout, LeakyReLU, Input, Activation, BatchNormalization
from keras.models import Model
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.optimizers import Adam
from keras.regularizers import l1, l2, l1_l2
from keras.datasets import mnist


'''importing specific modules for GANs'''

# from keras_adversarial.legacy import Dense,  BatchNormalization, Convolution2D
from keras_adversarial import AdversarialModel, ImageGridCallback, simple_gan, gan_targets
from keras_adversarial import AdversarialOptimizerSimultaneous, AdversarialOptimizerAlternating, normal_latent_sampling





def leaky_relu(n):
    '''
    Given a model with n targets and k players creates a model with n*k targets where each palyer optimizes loss on that players target
    :param n:
    :return:
    '''
    return k.relu(n, 0.2)

def gan_targets(n):
    '''
    [generator_fake, generator_real, discriminator_fake, discriminator_real]
    :param n: number of samples
    :return: number of samples
    '''
    generator_fake, generator_real = np.ones((n, 1)), np.zeros((n, 1))
    discriminator_fake, discriminator_real = np.zeros((n, 1)), np.ones((n, 1))
    return [generator_fake, generator_real, discriminator_fake, discriminator_real]

''' Defining generator, here each modeule in our pipeline is empty passed as input to the folloeing module'''
def model_generator():
    nch=256   # number of channels
    g_input = Input(shape=[100])
    H = Dense(nch*14*14, kernel_initializer='glorot_normal')(g_input)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)
    H = dim_ordering_reshape(nch, 14)(H)
    H = UpSampling2D(size=(2, 2))(H)


    # H = Conv2D(filters=nch, kernel_size=3, strides=(2,2))
    H = Conv2D(int(nch/2), 3, 3, padding='same', kernel_initializer='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)

    H = Conv2D(int(nch/4), 3, 3, padding='same', kernel_initializer='glorot_uniform')(H)
    H = BatchNormalization(mode=2)(H)
    H = Activation('relu')(H)


    H = Conv2D(1, 1, 1, padding='same', kernel_initializer='glorot_uniform')(H)
    H = Activation('sigmoid')(H)

    return Model(g_input, H)


'''Defining desriminator, here descriminator uses LeakyRelu'''
def model_descriminator(input_shape=(1, 28, 28), dropout_rate=0.5):
    nch=512
    d_input = dim_ordering_input(input_shape, name='input_x')

    H = Conv2D(int(nch/2), 5 ,5, strides=(2, 2), padding='same', activation='relu')(d_input)
    H = LeakyReLU(alpha=0.2)(H)

    H = Conv2D(nch, 5, 5, strides=(2, 2), padding='same', activation='relu')(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(rate=dropout_rate)(H)

    H = Flatten()(H)

    H = Dense(int(nch/2))(H)
    H = LeakyReLU(alpha=0.2)(H)
    H = Dropout(rate=dropout_rate)(H)

    d_v = Dense(1, activation='sigmoid')(H)
    return Model(d_input, d_v)

'''Defining two separate functions for loading and normalizing MNIST data'''
def mnist_process(x):
    return  x.astype(np.float32) / 255.0
def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

'''generator for creating new forged images at each epoch during training that looks like the original'''
def generator_samples(latent_dim, generator):
    def fun():
        z_samples = np.random.normal(size=(10*10, latent_dim))
        gen = dim_ordering_unfix(generator.predict(z_samples))
        return gen.reshape((10, 10, 28, 28))
    return fun()


'''Combining generator and descriminator to define a GANs'''

if __name__ == "__main__":
    path = '/kaggle/working/convoltional_gan'
    os.makedirs(path=path, exist_ok=True)
    latent_dim = 100  # z in R^100
    input_shape = (1, 28, 28)  # x in R^{28, 28}

    # generator { z -> x}
    generator = model_generator()

    # desriminaor { x -> y}
    descriminator = model_descriminator(input_shape=input_shape)

    # gan { x -> yfake, yreal}, z generated on GPU
    # weights are initialized with normal_latent_sampling which samples from a normal Gaussian distribution
    gan = simple_gan(generator=generator, discriminator=descriminator,
                     latent_sampling=normal_latent_sampling((latent_dim, )))

    # printing summary of models
    generator.summary()
    descriminator.summary()
    gan.summary()

    '''Building Adversarial model'''
    model = AdversarialModel(base_model=gan,
                             player_params=[generator.trainable_weights, descriminator.trainable_weights],
                             player_names=['generator', 'desriminator'])
    model.adversarial_compile(adversarial_optimizer=AdversarialOptimizerSimultaneous(),
                              player_optimizers=[Adam(learning_rate=1e-4, decay=1e-4), Adam(learning_rate=1e-3, decay=1e-4)],
                              loss='binary_crossentropy')

    '''train model'''
    generator_ob = ImageGridCallback(image_path=os.path.join(path, 'epoch-{03d}.png'),
                                     generator=generator_samples(latent_dim=latent_dim, generator=generator))

    xtrain, xtest = mnist_data()
    xtrain = dim_ordering_fix(xtrain.reshape((-1, 1, 28, 28)))
    xtest = dim_ordering_fix(xtest.reshape((-1, 1, 28, 28)))

    ytrain = gan_targets(xtrain.shape[0])
    ytest = gan_targets(xtest.shape[0])

    history = model.fit(x=xtrain, y=ytrain, validation_data=(xtest, ytest),
                        callbacks=[generator_ob], nb_epoch=100, batch_size=32)
    df = pd.DataFrame(history.history)
    df.to_csv(os.path.join(path, 'history.csv'))
    generator.save(os.path.join(path, 'generator.h5'))
    descriminator.save(os.path.join(path, 'descriminator.h5'))
