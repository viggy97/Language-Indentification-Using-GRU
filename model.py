import librosa
import os
import numpy as np
import pandas as pd
import glob, os
import argparse
import matplotlib.pyplot as plt
import shutil
from tensorflow.keras.layers import Input, Dense, GRU, Dropout
from tensorflow.keras import Model
import tensorflow
from tensorflow import keras

from tensorflow.keras import regularizers
from sklearn.utils import class_weight
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
import os
import librosa
import numpy as np
import pandas as pd
import glob, os
import argparse
import matplotlib.pyplot as plt
from torchsummary import summary

from sklearn.utils import class_weight


### DATA

num_feature = 64

class Dataset(Dataset):
    def __init__(self, num_seq, seq_length, num_feature,x,y):
        self.num_seq = num_seq
        self.seq_length = seq_length
        
        #x = np.random.randint(0, high=2, size = (num_seq * seq_length, num_feature))
        #x = np.sign(x - 0.5)
        
        #y = np.sum((x == np.roll(x, 1, axis = 0)), axis = 1)
        
        self.X = x.reshape(num_seq, seq_length, num_feature)
        self.Y = y.reshape(num_seq, 1)
        
        
    def __len__(self):
        return self.num_seq
    
    
    def __getitem__(self, index):
        # TEMP
        x = self.X[index:index+1]
        x = x.squeeze(0)
        
        y = self.Y[index]
       
        
        return torch.Tensor(x).float(), torch.Tensor(y).long()
        
 
train_num_seq =  315     # 16, demo
test_num_seq = 10

train_seq_length = 1000   # 4, demo

batch_size = 20


train_seq_eng = np.array([])
train_seq_hindi = np.array([])
train_seq_Mand = np.array([])

train_file = '/Users/shashank/Documents/Resumes/LatestResume/Coverletter/exam/DLHW5/train/'
for subdir, dirs, files in os.walk(train_file): 
    for file in files:
        print(file)
#         if file is not 'hindi_0004.wav' or 'hindi_0014.wav' or 'hindi_0016.wav' or 'mandarin_0014.wav':
        y, sr = librosa.load(subdir + "/" + file ,sr=16000)
        if 'train_english' in subdir: 
            #print(y)
            y = y[abs(y)>0.01]
            mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft =int(sr*0.025), hop_length = int(sr*0.010))
            preprocess_train_english = mat[:,0:1000].T
            train_seq_eng = np.append(train_seq_eng, preprocess_train_english)
        elif 'train_hindi' in subdir:
            y = y[abs(y)>0.01]
            mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft =int(sr*0.025), hop_length = int(sr*0.010))
            preprocess_train_hindi = mat[:,0:1000].T
            #
            train_seq_hindi = np.append(train_seq_hindi, preprocess_train_hindi)
        else:
            y = y[abs(y)>0.008]
            mat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=64, n_fft =int(sr*0.025), hop_length = int(sr*0.010))
            preprocess_train_Mand = mat[:,0:1000].T
            #mat[:,0:1000].T
            train_seq_Mand = np.append(train_seq_Mand, preprocess_train_Mand)

train_num_seq =  315 
#print(train_seq_eng.shape)
#print(train_seq_hindi.shape)
#print(train_seq_Mand.shape)
English_SEQ = np.reshape(train_seq_eng, [164,1000,64])
Labels_eng = np.zeros([164,1])
Hindi_SEQ = np.reshape(train_seq_hindi, [41,1000,64])
Labels_Hindi = np.ones([41,1])
Mand_SEQ = np.reshape(train_seq_Mand, [110,1000,64])
Labels_Mand = np.ones([110,1])*2

Train_data = np.vstack([English_SEQ,Hindi_SEQ, Mand_SEQ])
labels = np.vstack([Labels_eng, Labels_Hindi, Labels_Mand])
np.save('Train_data', Train_data)
np.save('Labels', labels)


Train_data = np.load('Train_data.npy')
#labels = np.load('Labels.npy')
Labels_eng = np.zeros([164,1000,1])
Labels_Hindi = np.ones([41,1000,1])
Labels_Mand = np.ones([110,1000,1])*2
labels = np.vstack([Labels_eng, Labels_Hindi, Labels_Mand])
print(labels.shape)

labelsf = np.reshape(labels, [315*1000, 1])
dataf = np.reshape(Train_data, [315*1000, 64])
rng_state = np.random.get_state()
np.random.shuffle(dataf)
np.random.set_state(rng_state)
np.random.shuffle(labelsf)
labels = np.reshape(labelsf, [315, 1000, 1])
Train_data = np.reshape(dataf, [315, 1000, 64])
X_train, X_test = train_test_split(Train_data,test_size=0.2)
y_train, y_test = train_test_split(labels,test_size=0.2)

lab = np.reshape(labels, [315*1000,])
class_weights = class_weight.compute_class_weight('balanced', np.unique(lab), lab)

training_in_shape = Train_data.shape[1:]
training_in = Input(shape = training_in_shape)
Var = GRU(1240, return_sequences=True, stateful = False)(training_in)
training_pred = Dense(3, activation = 'softmax')(Var)

training_model = Model(inputs = training_in, outputs = training_pred)
training_model.compile(loss = keras.losses.SparseCategoricalCrossentropy(),
                      optimizer = 'adam',
                      metrics = ['accuracy'])
training_model.summary()

results = training_model.fit(Train_data, labels, batch_size = 4, epochs = 20,
                            validation_split = 0.2)
                            

streaming_in = Input(batch_shape=(1,None,64))  ## stateful ==> needs batch_shape specified
foo = GRU(1240, return_sequences=False, stateful=True )(streaming_in)
streaming_pred = Dense(3, activation = 'softmax')(foo)
streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

streaming_model.compile(loss = keras.losses.SparseCategoricalCrossentropy(),
                      optimizer = 'adam',
                      metrics = ['accuracy'])
streaming_model.summary()


###### copy the weights from trained model to streaming-inference model
training_model.save_weights('weights.hdf5', overwrite=True)
streaming_model.load_weights('weights.hdf5')
keras.utils.plot_model(training_model, to_file='Streaming_Model.png', show_shapes=True, show_layer_names=True)

DEMO = 1
if DEMO:
    ##### demo the behaivor
    print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
    for s in range(1):
        print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
        for n in range(20):
            in_feature_vector = Train_data[s][n].reshape(1,1,64)
            single_pred = streaming_model.predict(in_feature_vector)[0]
            print(single_pred)
            streaming_model.reset_states()
                  
            p1 = plt.scatter(n, single_pred[0], color = 'blue')
            p2 = plt.scatter(n, single_pred[1], color = 'red')
            p3 = plt.scatter(n, single_pred[2], color = 'green')
    p1.set_label('English')
    p2.set_label('Hindi')
    p3.set_label('Mandarin')
    plt.savefig('model_plot.png')
    plt.legend()
    plt.show()
    

training_model.save('EE599_HW5_training.hdf5')

streaming_model = load_model('/content/drive/MyDrive/EE599_HW5_training.hdf5')

Num_sequences=1
Num_samples = 1000

DEMO = 1
if DEMO:
    ##### demo the behaivor
    print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
    for s in range(Num_sequences):
        print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
        for n in range(Num_samples):
            in_feature_vector = X_test[s][n].reshape(1,1,64)
            single_pred = streaming_model.predict(in_feature_vector)[0]
            print(single_pred)
            streaming_model.reset_states()#
