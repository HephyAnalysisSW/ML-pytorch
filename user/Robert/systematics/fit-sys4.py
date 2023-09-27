#!/usr/bin/env python
import ROOT, os
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('../../..')

import tools.syncer as syncer
import tools.user as user 

c1 = ROOT.TCanvas() # do this to avoid version conflict in png.h with keras import ...
c1.Draw()
c1.Print('/tmp/delete.png')

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--prefix',             action='store', type=str,   default='v1', help="Name of the training")
argParser.add_argument('--epochs',             action='store', type=int,   default=100, help="Number of epochs")
argParser.add_argument('--small',              action='store_true', help="Small?")
argParser.add_argument('--output_directory',   action='store', type=str,   default='/eos/vbc/group/cms/robert.schoefbeck/tt-jec/models/')
args = argParser.parse_args()

# fix random seed for reproducibility
np.random.seed(1)

#########################################################################################
# Training data 
import data_models.TTLep_pow_sys as data_model 
generator = data_model.DataGenerator(maxN=1000 if args.small else None)

# directories
plot_directory   = os.path.join( user.plot_directory, 'tt-jec', 'training', args.prefix )
output_directory = os.path.join( args.output_directory, args.prefix) 

#########################################################################################
# define model (neural network)
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.layers import Input, Activation, Dense, Convolution2D, MaxPooling2D, Dropout, Flatten, Concatenate
from keras.layers import BatchNormalization
#from keras.utils import np_utils

# flat layers
n_var_flat = len(data_model.feature_names)
inputs = Input(shape=(n_var_flat, ))
x = BatchNormalization(input_shape=(n_var_flat, ))(inputs)
x = Dense(n_var_flat*2, activation='LeakyReLU')(x)
x = Dense(n_var_flat+5, activation='LeakyReLU')(x)
x = Dense(n_var_flat+5, activation='LeakyReLU')(x)

outputs = Dense(2, kernel_initializer='normal', activation=None)(x)
model = Model( inputs, outputs )

def loss(truth, prediction):

    mask_nominal        = tf.squeeze( (truth==0) )
    mask_nominal.set_shape([None])
    prediction_nominal  = tf.boolean_mask(prediction, mask_nominal)

    loss = 0
    for level in generator.levels:
        #print ("Adding loss level", level)
        if level==0: continue #sanity

        mask_level       = tf.squeeze( (truth==level) )
        mask_level.set_shape([None])
        prediction_level = tf.boolean_mask(prediction, mask_level)

        #print("shape", mask_nominal.shape, mask_level.shape, prediction.shape, "level", level)       
 
        loss += tf.reduce_sum(tf.math.log(1+tf.math.exp(   level*prediction_nominal[:,0] + level**2*prediction_nominal[:,1])))
        loss += tf.reduce_sum(tf.math.log(1+tf.math.exp( - level*prediction_level[:,0]   - level**2*prediction_level[:,1])))

        #if not loss>0:
        #    print ("smth wrong with loss", loss)
        #    print (level, prediction_nominal[:,0], tf.math.exp(-level*prediction_level[:,0]) )

    return loss 

#model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
opt = Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=loss)
model.summary()

# define callback for early stopping
import tensorflow as tf
tf.debugging.disable_traceback_filtering()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) # patience can be higher if a more accurate result is preferred

from sklearn.model_selection import train_test_split

x,y = generator[0]

x_train, x_test, y_train, y_test = train_test_split(x, y)

del x, y

x_train = tf.where(tf.math.is_nan(x_train), 0., x_train)
y_train = tf.where(tf.math.is_nan(y_train), 0., y_train)
x_test = tf.where(tf.math.is_nan(x_test), 0., x_test)
y_test = tf.where(tf.math.is_nan(y_test), 0., y_test)

# train the model
history = model.fit(
                    x=x_train,y=y_train, #*(generator[0]),
                    epochs=args.epochs, 
                    verbose=1, # switch to 1 for more verbosity, 'silences' the output
                    validation_data=(x_test,y_test),
                    #validation_data = validation_data_generator,
                    callbacks=[callback],
                    shuffle=False,
                    #batch_size=32,
                   )
print('training finished')

# saving
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

output_file = os.path.join(output_directory, 'multiclass_model.h5')
if os.path.exists(output_file):
    os.remove( output_file)
model.save(output_file)
print("Written model to: %s"% output_file)
