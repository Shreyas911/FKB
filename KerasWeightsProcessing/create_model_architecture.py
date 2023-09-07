# adapted from https://github.com/scientific-computing/FKB/blob/master/GettingStarted.ipynb

import os
import numpy as np

import tensorflow as tf
import tensorflow.keras as keras

from tensorflow import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Activation


dir_mod = '/home/shreyas/FKB/models/'
lab_mod = 'model_3layers_20210830'
modh5   = os.path.join(dir_mod,lab_mod + '.h5')
modtxt  = os.path.join(dir_mod,lab_mod + '.txt')

# build model
keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = Sequential()
# input: along x the variables in lab_sea verification experiments are of size 18
# I am concatenating uice and vice, so my input will be 36
# I will also need an output of 36
# model.add(Dense(8,  input_dim=36, activation='relu'))
# model.add(Dense(12, activation='relu'))
# model.add(Dense(36,  activation='linear'))

model.add(Dense(8,  input_dim=5, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(5,  activation='linear'))

#model.add(Dense(8,  input_dim=18,activation='relu'))
#model.add(Dense(12, activation='relu'))
#model.add(Dense(2,  activation='linear'))

model.compile(loss='mse', optimizer='SGD')
model.summary()


# save model
model.save(modh5)

# convert to Fortran-accepted format
import sys
sys.path.append('/home/shreyas/FKB')
from KerasWeightsProcessing.convert_weights import txt_to_h5, h5_to_txt

h5_to_txt(
    weights_file_name=modh5, 
    output_file_name=modtxt
)

# examine the file
#!cat getting_started_model.txt

