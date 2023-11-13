#!/usr/bin/env python
import ROOT, os
ROOT.gStyle.SetOptStat(0)
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT.gROOT.LoadMacro(os.path.join( dir_path, "../tools/scripts/tdrstyle.C"))
ROOT.setTDRStyle()
import numpy as np
if __name__=="__main__":
    import sys
    sys.path.append('..')

import tools.syncer as syncer
import tools.user as user 
import tools.helpers as helpers 
import tensorflow as tf
#tf.keras.backend.set_floatx('float64')

#import tensorflow_probability as tfp
c1 = ROOT.TCanvas() # do this to avoid version conflict in png.h with keras import ...
c1.Draw()
c1.Print('/tmp/delete.png')

import argparse
argParser = argparse.ArgumentParser(description = "Argument parser")
argParser.add_argument('--prefix',             action='store', type=str,   default='v1', help="Name of the training")
argParser.add_argument('--data_model',         action='store', type=str,   default='TTLep_pow_sys', help="Which data model?")
argParser.add_argument('--epochs',             action='store', type=int,   default=100, help="Number of epochs")
argParser.add_argument('--small',              action='store_true', help="Small?")
argParser.add_argument('--quadratic',              action='store_true', help="quadratic?")
argParser.add_argument('--overwrite',          action='store_true', help="Overwrite?")
argParser.add_argument('--output_directory',   action='store', type=str,   default=os.path.join(user.model_directory,'tt-jec/models/') )
args = argParser.parse_args()

# fix random seed for reproducibility
np.random.seed(1)

#########################################################################################
# Training data 

exec( "import data_models.%s as data_model"%args.data_model )

generator = data_model.DataGenerator(maxN=200000 if args.small else None)#, levels = levels)

# directories
plot_directory   = os.path.join( user.plot_directory, 'tt-jec', args.data_model, args.prefix+('_small' if args.small else "") + ("_quadratic" if args.quadratic else ""), 'training')
output_directory = os.path.join( args.output_directory, args.data_model, args.prefix+('_small' if args.small else "") + ("_quadratic" if args.quadratic else "")) 

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
x = Dense(n_var_flat*2, activation='sigmoid')(x)
x = Dense(n_var_flat+5, activation='sigmoid')(x)
#x = Dense(n_var_flat+5, activation='sigmoid')(x)

outputs = Dense(2 if args.quadratic else 1, kernel_initializer='normal', activation=None)(x)
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

        nom = level*prediction_nominal[:,0]
        lev = level*prediction_level[:,0]
        if args.quadratic:
            nom += level**2*prediction_nominal[:,1]
            lev += level**2*prediction_level[:,1]

        loss += tf.reduce_sum(-tf.math.log(2.) + tf.math.softplus(nom))
        loss += tf.reduce_sum(-tf.math.log(2.) + tf.math.softplus(-lev))

        #loss += tf.reduce_sum( 1./(1+tf.math.exp( nom ) )**2 )
        #loss += tf.reduce_sum( 1./(1+tf.math.exp( -lev ) )**2 )

        #loss += tf.reduce_sum(tf.math.log((1+tf.math.exp(   exponent_nom )))) #FIXME this is the worst implementation
        #loss += tf.reduce_sum(tf.math.log((1+tf.math.exp( - exponent_lev ))))

        #tf.print()
        #tf.print("level", level, nom, lev, tf.reduce_sum( 1./(1+tf.math.exp( nom ) )**2 ), tf.reduce_sum( 1./(1+tf.math.exp( -lev ) )**2 ))
        #tf.print("level", level, prediction_nominal.shape, prediction_nominal[:,0], tf.math.log(0.5*(1+tf.math.exp(   level*prediction_nominal[:,0] ))))
        #tf.print("level", level, prediction_level.shape, prediction_level[:,0], tf.math.log(0.5*(1+tf.math.exp( - level*prediction_level[:,0] ))))

        #loss += tf.reduce_sum(    tf.math.exp(   2*(level*prediction_nominal[:,0] + level**2*prediction_nominal[:,1])) )
        #loss += tf.reduce_sum( (1-tf.math.exp( level*prediction_level[:,0] +level**2*prediction_level[:,1]))**2 )

        #if not loss>0:
        #    print ("smth wrong with loss", loss)
        #    print (level, prediction_nominal[:,0], tf.math.exp(-level*prediction_level[:,0]) )

    return loss 

from sklearn.model_selection import train_test_split

features, variations = generator[0]
features_train, features_test, variations_train, variations_test = train_test_split(features, variations)

#predictions = np.ones((len(variations_train),1))

opt = Adam(learning_rate=0.001)
#opt = SGD(learning_rate=0.01)
model.compile(optimizer=opt, loss=loss)
model.summary()

# define callback for early stopping
tf.debugging.disable_traceback_filtering()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3) # patience can be higher if a more accurate result is preferred

#del x, y

count_0 = np.count_nonzero(variations_train==0)
count_1 = np.count_nonzero(variations_train==1)
print ("Training with", count_0,count_1, "expect", np.log(count_1/count_0), "rel stat error", np.sqrt(1./count_0+1./count_1)) 

output_file = os.path.join(output_directory, 'multiclass_model.h5')
if not os.path.exists(output_file) or args.overwrite:

    features_train    = tf.where(tf.math.is_nan(features_train), 0., features_train)
    variations_train  = tf.where(tf.math.is_nan(variations_train), 0., variations_train)
    features_test     = tf.where(tf.math.is_nan(features_test), 0., features_test)
    variations_test   = tf.where(tf.math.is_nan(variations_test), 0., variations_test)

    # train the model
    history = model.fit(
                        x=features_train,y=variations_train, #*(generator[0]),
                        epochs=args.epochs, 
                        verbose=1, # switch to 1 for more verbosity, 'silences' the output
                        #validation_data=(features_test,variations_test),
                        validation_data=(features_train,variations_train), #FIXME!!!
                        #validation_data = validation_data_generator,
                        callbacks=[callback],
                        shuffle=False,
                        #batch_size=32,
                       )
    print('training finished.')

    # saving
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    if os.path.exists(output_file):
        os.remove( output_file)
    model.save(output_file)
    print("Written model to: %s"% output_file)
else:
    model.load_weights(output_file)
    print("Loaded model from  %s"% output_file)

_prediction  = model(features[variations[:,0]==0]).numpy()

print ("Mean prediction", _prediction.mean())
def prediction( nu ):
    return np.exp( nu*_prediction[:,0] + ( nu**2*_prediction[:,1] if args.quadratic else 0))
#def prediction( nu ):
#    return np.exp( -nu*_prediction[:,0] - nu**2*_prediction[:,1]-1)

color = {2:ROOT.kRed, 1.5: ROOT.kOrange+10, 1:ROOT.kMagenta+2, 0.5: ROOT.kBlue+1, -0.5: ROOT.kCyan+2, -1.:ROOT.kGreen+2, -1.5: ROOT.kYellow+2, -2:ROOT.kAzure-8} 
for i_feature, feature in enumerate(data_model.feature_names):

    c1 = ROOT.TCanvas()
    l = ROOT.TLegend(0.2,0.8,0.9,0.95)
    l.SetNColumns(3)
    l.SetFillStyle(0)
    l.SetShadowColor(ROOT.kWhite)
    l.SetBorderSize(0)

    binning   = data_model.plot_options[feature]['binning'] 
    np_binning= np.linspace(binning[1], binning[2], 1+binning[0])

    h_nominal  = helpers.make_TH1F(np.histogram(features[variations[:,0]==0][:,i_feature], np_binning ))
    h_nominal  .SetLineColor(ROOT.kGray+2)
    h_nominal  .SetMarkerColor(ROOT.kGray+2)
    h_nominal  .SetMarkerStyle(0)

    h_nominal.Draw()
    h_nominal.SetTitle("")
    h_nominal.GetXaxis().SetTitle(data_model.plot_options[feature]['tex'])
    c1.SetLogy()
    l.AddEntry(h_nominal, "nominal")
    h_level_truth = {}
    h_level_pred  = {}
    for level in generator.levels:
            
        h_level_truth[level]  = helpers.make_TH1F(np.histogram(features[variations[:,0]==level][:,i_feature], np_binning))
        h_level_truth[level] .SetLineColor(color[level])
        h_level_truth[level].SetLineStyle(ROOT.kDashed)
        h_level_truth[level]      .SetMarkerColor(color[level])
        h_level_truth[level]      .SetMarkerStyle(0)
        h_level_pred[level]  = helpers.make_TH1F(np.histogram(features[variations[:,0]==0][:,i_feature], np_binning, weights=prediction(level)))
        h_level_pred[level] .SetLineColor(color[level])
        h_level_pred[level]      .SetMarkerColor(color[level])
        h_level_pred[level]      .SetMarkerStyle(0)

        #l.AddEntry(h_level_truth[level]   , "#sigma = %2.1f (truth)"%level )
        l.AddEntry(h_level_pred[level]   , "#sigma = %2.1f"%level )

        h_level_truth[level].Draw("same")
        h_level_pred[level].Draw("same")

    h_nominal.Draw("same")

    c1.Print(os.path.join( plot_directory, feature+'.png'))

    ref = h_nominal.Clone()

    h_nominal.Divide(ref)
    h_nominal.Draw()
    c1.SetLogy(0)
    h_nominal.GetYaxis().SetRangeUser(0.8,1.4)
    for level in generator.levels:
        h_level_truth[level].Divide(ref)
        h_level_pred[level].Divide(ref)
        h_level_truth[level].Draw("same")
        h_level_pred[level].Draw("same")
    l.Draw()
    c1.Print(os.path.join( plot_directory, "variation_"+feature+'.png'))

helpers.copyIndexPHP( plot_directory )
syncer.sync()
