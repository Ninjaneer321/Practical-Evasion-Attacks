# numpy stack
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

# keras
#from keras.callbacks import EarlyStopping, ModelCheckpoint
#from keras.layers import *
#from keras.models import *
#from keras import optimizers
#from tensorflow.keras import optimizers
#from keras.callbacks import *

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import *
from tensorflow import keras

# sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, precision_score, confusion_matrix, recall_score
from sklearn.preprocessing import MinMaxScaler
from autoencoder import load_AEED

# os and time utils
import os
import time
import glob

import pickle

import json

def create_dataset(dataset, window_size = 4):
    """ 
    Creates the dataset composed by window_size samples of sensor readings and their relative label
    
    if windows size is 2, it returns a dataset composed [[[x-1, x][x]], [[x, x+1][x+1]], ...]
    Parameters
    ----------
    dataset :  list
        list of dataset samples
    window_size : int
        number of samples used for feeding the network.

    Returns
    -------
    np array
        dataset samples organized in groups of windows_size
    np array
        target of model prediction
    """
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size + 1)] #remove +1 to turn into 1-step ahead prediction
        data_X.append(a)
        data_Y.append(dataset[i + window_size])
    return(np.array(data_X), np.array(data_Y))

def preprocess_physical(path):
    a_pd = pd.read_csv(path+'/scada_values.csv', parse_dates=['timestamp'] )
    a_pd = a_pd.dropna()

    # We drop rows with Bad input values
    for column in a_pd.columns:
        a_pd = a_pd.drop(a_pd[a_pd[column] == 'Bad Input'].index)

    alarms = [col for col in a_pd.columns if 'AL' in col]

    for alarm in alarms:
        exp = (a_pd[alarm] == 'Inactive')
        a_pd.loc[exp, alarm] = 0

        exp = (a_pd[alarm] == 'Active')
        a_pd.loc[exp, alarm] = 1

    return a_pd

def compute_scores(Y,Yhat, a_window, a_theta):
    fpr, recall, _ = roc_curve(Y, Yhat)
    return [accuracy_score(Y,Yhat),f1_score(Y,Yhat),precision_score(Y,Yhat),recall[1], fpr[1], a_window, a_theta]

def train_and_validate_encoder(X1, X1_prime, X2, X2_prime, this_nh, this_cf, activation='tanh', learning_rate=0.001):
    params = {
        'nI' : X.shape[1],
        'nH' : this_nh,
        'cf' : this_cf,
        'activation' : activation,
        'verbose' : 1,
        'learning_rate' : learning_rate
    }
    
    # create AutoEncoder for Event Detection (AEED)
    autoencoder = AEED(**params)
    autoencoder.initialize()

    # train models with early stopping and reduction of learning rate on plateau
    earlyStopping= EarlyStopping(monitor='val_loss', patience=3, verbose=0,  min_delta=1e-4, mode='auto')
    lr_reduced = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, verbose=0, epsilon=1e-4, mode='min')

    # initialize time
    start_time = time.time()

    # train autoencoder
    training_object = autoencoder.train(X1, X1_prime,
                epochs=500,
                batch_size=32,
                shuffle=True,
                callbacks = [earlyStopping, lr_reduced],
                verbose = 2,
                validation_data=(X2, X2_prime))  
    
    _, validation_errors = autoencoder.predict(X2, X2_prime)
    
    results = {
        'nh' : this_nh,
        'cf' : this_cf,
        'validation_errors' : validation_errors.mean(axis=0)
    }
    
    return results

# classes
class AutoEncoder(object):
    """ Keras-based AutoEncoder (AE) class used for event detection.

        Attributes:
        params: dictionary with parameters defining the AE structure,
    """
    def __init__(self, **kwargs):
        """ Class constructor, stores parameters and initialize AE Keras model. """
        
        # Default parameters values. If nI is not given, the code will crash later.
        params = {
            'nI': None,
            'nH': 3,
            'cf': 1,
            'activation' : 'tanh',
            'optimizer' : None,
            'verbose' : 0
            }

        for key,item in kwargs.items():
            params[key] = item
        
        self.params = params

    def create_model(self):
        """ Creates Keras AE model.

            The model has nI inputs, nH hidden layers in the encoder (and decoder)
            and cf compression factor. The compression factor is the ratio between
            the number of inputs and the innermost hidden layer which stands between
            the encoder and the decoder. The size of the hidden layers between the 
            input (output) layer and the innermost layer decreases (increase) linearly
            according to the cg.
        """
        
        # retrieve params
        nI = self.params['nI'] # number of inputs
        nH = self.params['nH'] # number of hidden layers in encoder (decoder)
        cf = self.params['cf'] # compression factor
        activation = self.params['activation'] # autoencoder activation function
        optimizer = self.params['optimizer'] # Keras optimizer
        verbose = self.params['verbose'] # echo on screen
        
        # get number/size of hidden layers for encoder and decoder
        temp = np.linspace(nI,nI/cf,nH + 1).astype(int)
        nH_enc = temp[1:]
        nH_dec = temp[:-1][::-1]

        # input layer placeholder
        input_layer = Input(shape=(nI,))

        # build encoder
        for i, layer_size in enumerate(nH_enc):
            if i == 0:
                # first hidden layer
                encoder = Dense(layer_size, activation=activation)(input_layer)
            else:
                # other hidden layers
                encoder = Dense(layer_size, activation=activation)(encoder)

        # build decoder
        for i, layer_size in enumerate(nH_dec):
            if i == 0:
                # first hidden layer
                decoder = Dense(layer_size, activation=activation)(encoder)
            else:
                # other hidden layers
                decoder = Dense(layer_size, activation=activation)(decoder)

        # create autoencoder
        autoencoder = Model(input_layer, decoder)
        if optimizer == None:
            try:
                optimizer = optimizers.Adam(lr = 0.001)
            except:
                optimizer = optimizers.Adam(learning_rate = 0.001)
        # print autoencoder specs
        if verbose > 0:
            print('Created autoencoder with structure:');
            print(', '.join('layer_{}: {}'.format(v, i) for v, i in enumerate(np.hstack([nI,nH_enc,nH_dec]))))

        # compile and return model
        autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')
        autoencoder.summary()
        return autoencoder
    
    def build_predictor(self):
        model = Sequential()
        '''
        model.add(LSTM(43,dropout_U = 0.2, dropout_W = 0.2, input_shape=(2,43)))# return_sequences=True, 
        #model.add(LSTM(43, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(Dense(43, activation = 'relu'))
        '''
        model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape = (1,39), padding='same'))
        model.add(MaxPooling1D(pool_size=1, strides=None))
        model.add(Conv1D(128, 2, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Conv1D(256, 2, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=1))
        model.add(Flatten())
        model.add(Dropout(0.3, noise_shape=None, seed=13))
        model.add(Dense(39, activation='relu'))     
        
        model.compile(loss='mean_squared_error', optimizer=  optimizers.Adam(lr = 0.001))
        model.summary()
        return model

    def train(self, x, y, **train_params):
        """ Train autoencoder,

            x: inputs (inputs == targets, AE are self-supervised ANN).
        """        
        if self.params['verbose']:
            if self.ann == None:
                print('Creating model.')
                self.create_model()
        self.ann.fit(x, y, **train_params)


    def predict(self, x, test_params={}):
        """ Yields reconstruction error for all inputs,

            x: inputs.
        """
        return self.ann.predict(x, **test_params)

class AEED(AutoEncoder):
    """ This class extends the AutoEncoder class to include event detection
        functionalities.
    """
    
    def difference(x):
        return (x[-1] - x[0])**2
    
    def initialize(self):
        """ Create the underlying Keras model. """
        self.ann = self.create_model() #build_predictor()#

    def predict(self, x, y, **keras_params):
        """ Predict with autoencoder. """        
        preds = super(AEED, self).predict(x,keras_params)
        errors = pd.DataFrame((y-preds)**2)
        return preds, errors        
        
    def detect(self, x, y, theta, window = 1, average=False, sys_theta = 0, **keras_params):
        """ Detection performed based on (smoothed) reconstruction errors.

            x = inputs,
            theta = threshold, attack flagged if reconstruction error > threshold,
            window = length of the smoothing window (default = 1 timestep, i.e. no smoothing),
            average = boolean (default = False), if True the detection is performed
                on the average reconstruction error across all outputs,
            keras_params = parameters for the Keras-based AE prediction.
        """
        #        preds = super(AEED, self).predict(x,keras_params)
        preds, temp = self.predict(x, y, **keras_params)
        #temp = (x-preds)**2
        if average:
            errors = temp.mean(axis=1).rolling(window=window).mean()             
            detection = errors > theta
        else:
            errors = temp.rolling(window=window).mean()
            detection = errors.apply(lambda x: x>np.max(theta.name, sys_theta)) 
            
        return detection, errors

    def save(self, filename, scaler, theta):
        """ Save AEED modelself.

            AEED parameters saved in a .json, while Keras model is stored in .h5 .
        """
        # parameters
        with open(filename+'.json', 'w') as fp:
            json.dump(self.params, fp)
        # keras model
        self.ann.save(filename+'.h5')
        with open("theta", 'w') as f:
            f.write(str(theta))
        pickle.dump(scaler, open( "scaler.p", "wb" ))
        # echo
        print('Saved AEED parameters to {0}.\nKeras model saved to {1}'.format(filename+'.json', filename+'.h5'))


# functions
def load_AEED(params_filename, model_filename):
    """ Load stored AEED. """
    # load params and create AEED
    with open(params_filename) as fd:
        params = json.load(fd)
    aeed = AEED(**params)

    # load keras model
    aeed.ann = load_model(model_filename)
    return aeed

def plot_detection(Y, Yhats, labels, filename='conv_autoencoder_phy_detections'):
    
    font = {'weight' : 'normal',
            'size'   : 18}
    plt.rc('font', **font)

    shade_of_gray = '0.75'
    n_subplots = len(Y)

    #ticks_labels = ['09:00am', '10:00am', '11:00am', '12:00m', '01:00pm', '02:00pm', '03:00pm', '04:00pm', '05:00pm']
    
    fig, axes = plt.subplots(n_subplots, figsize=(18, 50), dpi=80)    
    
    i = 0
    for i in range(0,n_subplots):    
        axes[i].plot(Yhats[i], color = shade_of_gray, label = 'Predicted state')
        axes[i].fill_between(Yhats[i].index, Yhats[i].values, where=Yhats[i].values <=1, interpolate=True, color=shade_of_gray)
        axes[i].plot(Y[i], color = 'r', alpha = 0.85, lw = 1, label = 'Real state')
        axes[i].set_title(labels[i], fontsize = 18)
        
        #ticks = np.arange(0, len(detections[i]), step=int(len(detections[i])/8))    
        #axes[i].set_xticks(ticks)
        #axes[i].set_xticklabels(ticks_labels)
        
        axes[i].set_yticks([0,1])
        axes[i].set_yticklabels(['NO ATTACK','ATTACK'])
        axes[i].legend(fontsize = 18, loc = 2)
            
    #ax.legend(fontsize = 18, loc = 2)
    plt.savefig(filename+'.png', orientation='landscape')
    plt.savefig(filename+'.pdf', orientation='landscape')    