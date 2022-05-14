from google.colab import drive
drive.mount('/content/drive')
import scipy
from scipy import io

%cd '/content/drive/My Drive/PostDocProject/keras-gcn/'

!python setup.py install
!pip install keras==1.1 tensorflow-gpu==0.12.0

from __future__ import print_function
import tensorflow
import keras
from keras.layers import Input, Dropout, Embedding, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Lambda, Concatenate, LayerNormalization, Add
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2

import kegra
from kegra.layers.graph import GraphConvolution
from kegra.utils import *

import time
import tensorflow as tf
from config import *

support=1


########################################
#  prepare  data                       #
########################################

# choose the appropriate flag 

if flag_dataset==0:
	davis_data=io.loadmat('/content/drive/My Drive/davis.mat')
	train_drugs = davis_data['train_drugs']
	train_prots = davis_data['train_prots']
	train_Y = davis_data['train_Y']

	val_drugs = davis_data['val_drugs']
	val_prots = davis_data['val_prots']
	val_Y = davis_data['val_Y']
elif flag_dataset==1:
    kiba_data=io.loadmat('/content/drive/My Drive/KIBA.mat')
	train_drugs = kiba_data['train_drugs']
	train_prots = kiba_data['train_prots']
	train_Y = kiba_data['train_Y']

	val_drugs = kiba_data['val_drugs']
	val_prots = kiba_data['val_prots']
	val_Y = kiba_data['val_Y']
else:
    bindingdb_data=io.loadmat('/content/drive/My Drive/KIBA.mat')
	train_drugs = bindingdb_data['train_drugs']
	train_prots = bindingdb_data['train_prots']
	train_Y = bindingdb_data['train_Y']

	val_drugs = bindingdb_data['val_drugs']
	val_prots = bindingdb_data['val_prots']
	val_Y = bindingdb_data['val_Y']

train_Y = np.reshape(train_Y,(train_Y.shape[1],))
val_Y = np.reshape(val_Y,(val_Y.shape[1],))


num_train_samples = train_Y.shape[0]
num_val_samples = val_Y.shape[0]
###################################################


# Define parameters
DATASET = 'cora'
FILTER = 'localpool'  # 'chebyshev'
MAX_DEGREE = 2  # maximum polynomial degree
SYM_NORM = True  # symmetric (True) vs. left-only (False) normalization
NB_EPOCH = 200
PATIENCE = 10  # early stopping patience


def cindex_score(y_true, y_pred):

    g = tf.subtract(tf.expand_dims(y_pred, -1), y_pred)
    g = tf.cast(g == 0.0, tf.float32) * 0.5 + tf.cast(g > 0.0, tf.float32)

    f = tf.subtract(tf.expand_dims(y_true, -1), y_true) > 0.0
    f = tf.compat.v1.matrix_band_part(tf.cast(f, tf.float32), -1, 0)

    g = tf.reduce_sum(tf.multiply(g, f))
    f = tf.reduce_sum(f)

    return tf.where(tf.equal(g, 0), 0.0, g/f) #select

def dot_batch(inp):
    import tensorflow as tf
    return tf.math.sigmoid(tf.tensordot(inp[0][0:100],inp[1],axes=(1, 1)))

def dot_batch2(inp):
    import tensorflow as tf
    a=inp[0]
    b=inp[1]
    return tf.tensordot(a,b,axes=(0, 0))

# Compound Feature Extractor


Drug_input = Input(shape=(smiles_max_len,), dtype='int32',name='drug_input') 
Protein_input = Input(shape=(protein_max_len,), dtype='int32',name='protein_input')
G = [Input(batch_shape=(None, None))]#, batch_shape=(None, None))]

encode_smiles = Embedding(input_dim=smiles_dict_len+1, output_dim = embedding_size, input_length=smiles_max_len,name='smiles_embedding')(Drug_input) 
encode_smiles = Conv1D(filters=num_filters, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_smiles')(encode_smiles)
encode_smiles = Conv1D(filters=num_filters*2, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_smiles')(encode_smiles)
encode_smiles = Conv1D(filters=num_filters*3, kernel_size=smiles_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_smiles')(encode_smiles)


    
encode_protein = Embedding(input_dim=protein_dict_len+1, output_dim = embedding_size, input_length=protein_max_len, name='protein_embedding')(Protein_input)
encode_protein = Conv1D(filters=num_filters, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv1_prot')(encode_protein)
encode_protein = Conv1D(filters=num_filters*2, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv2_prot')(encode_protein)
encode_protein = Conv1D(filters=num_filters*3, kernel_size=protein_filter_lengths,  activation='relu', padding='valid',  strides=1, name='conv3_prot')(encode_protein)
    
encode_protein = GlobalMaxPooling1D()(encode_protein)
encode_smiles = GlobalMaxPooling1D()(encode_smiles)



protein_desc_len=96


X_in =Concatenate()([encode_protein,encode_smiles])  # this correspond with node descriptor
X_in= LayerNormalization()(X_in)

model_Feat = Model(inputs=[Drug_input, Protein_input], outputs=[X_in])

input_adj = Input(shape=(num_filters*6,))
FC1 = Dense(1024, activation='relu', name='dense1')(input_adj)
FC2 = Dropout(0.1)(FC1)
FC2 = Dense(1024, activation='relu', name='dense2')(FC2)
FC2 = Dropout(0.1)(FC2)
FC2 = Dense(512, activation='relu', name='dense3')(FC2)
# And add a logistic regression on top
G1 = Dense(1, kernel_initializer='normal',  name='dense4')(FC2) # if you want train model for active/inactive set activation='sigmoid'

model_Adj = Model(inputs=input_adj, outputs=[G1])


H = Dropout(0.5)(X_in)
H = GraphConvolution(512, support, activation='relu', kernel_regularizer=l2(5e-4))([X_in]+G) #, kernel_regularizer=l2(5e-4)
#H = Dropout(0.5)(H)
#H = GraphConvolution(192, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)
H = GraphConvolution(256, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G)

Y = GraphConvolution(192, support, activation='relu', kernel_regularizer=l2(5e-4))([H]+G) #activation='sigmoid',, name='main_out'

Y = Add()([Y,X_in])
Y = Dense(1024, activation='relu')(Y)
Y = Dropout(0.1)(Y)
Y = Dense(1024, activation='relu')(Y)
Y = Dropout(0.1)(Y)
Y = Dense(512, activation='relu')(Y)
# And add a logistic regression on top
Y = Dense(1, kernel_initializer='normal')(Y)

model_Semi = Model(inputs=[Drug_input, Protein_input,G], outputs=Y)

# Compile model

METRICS = [
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR') # precision-recall curve
]

def weighted_bce(y_true, y_pred):
  y_true = keras.backend.cast(y_true,"float32")
  weights = (y_true * 18.) + 1.
  bce = keras.backend.binary_crossentropy(y_true, y_pred)
  weighted_bce = keras.backend.mean(bce * weights)
  return weighted_bce
def bce(y_true, y_pred):
  y_true = keras.backend.cast(y_true,"float32")
  
  bce = keras.backend.binary_crossentropy(y_true[0,:], y_pred[0,:])
  weighted_bce = keras.backend.mean(bce)
  return weighted_bce

out_ = model_Adj(model_Feat([Drug_input,Protein_input]))
model2 = Model(inputs=[Drug_input,Protein_input], outputs=out_)
model2.compile(loss='mse', optimizer=Adam(learning_rate=learning_rate_feat_model),metrics=['mse',cindex_score]) #weighted_bce


import keras.backend as K
def mean_squared_error_w(y_true, y_pred):
    mses = K.mean(K.square(y_pred - y_true), axis = -1)
    std_of_pred = K.std(y_pred)
    std_of_true = K.std(y_true)
    mses_std =K.square(std_of_pred - std_of_true)
    #const = K.mean(mses, axis = -1) + (std_of_mses * 0.5)
    #mask = K.cast(K.less(K.mean(K.square(y_pred - y_true), axis=-1), const), dtype = "float32")
    return mses_std+mses
def mean_squared_error_w1(y_true, y_pred):
    mses = K.mean(K.square(y_pred - y_true), axis = -1)
    std_of_mses = K.std(mses)
    const = K.mean(mses, axis = -1) + (std_of_mses * 0.5)
    mask = K.cast(K.less(K.mean(K.square(y_pred - y_true), axis=-1), const), dtype = "float32")
    return mask * K.mean(K.square(y_pred - y_true), axis=-1)


model_Semi.compile(loss=['mse'], optimizer=Adam(learning_rate=learning_rate_GCN),metrics=['mse',cindex_score])
# Helper variables for main training loop
wait = 0
preds = None
best_val_loss = 99999



###########################################
#  Design Generator                       #
###########################################
from scipy.sparse import csr_matrix
def generate_data(batch_size,flag):
  i_c=0
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0
        
        if i_c>=num_train_samples:
          i_c=0

        in_drug=train_drugs[i_c:i_c+batch_size]
        in_prots=train_prots[i_c:i_c+batch_size]

        idx = np.concatenate((range(0,i_c),range(i_c+1+batch_size,train_drugs.shape[0])))
        idx=idx.astype(int)
        
        pindex = np.random.choice(idx.shape[0], 10*batch_size, replace=False)  

        input1 = np.concatenate((in_drug,train_drugs[idx[pindex]]), axis=0)
        input2 = np.concatenate((in_prots,train_prots[idx[pindex]]), axis=0)

        output1 = np.concatenate((train_Y[i_c:i_c+batch_size],train_Y[idx[pindex]]), axis=0)
        output1 = np.reshape(output1,(10*batch_size+batch_size,1))
  
        G = model2.predict([np.array(input1[0:batch_size]),np.array(input2[0:batch_size])],batch_size=batch_size)
        
        output1_ = (output1>threshold)
        output1_ = output1_.astype(int)
        tmpp=np.matmul(output1_,np.transpose(output1_))+np.matmul(1-output1_,np.transpose(1-output1_))
        
        
        G1=np.array(G)
        G1 = (G1>threshold)
        G1 = G1.astype(int)
        
        G1 = np.matmul(G1,np.transpose(output1_))+np.matmul(1-G1,np.transpose(1-output1_))
        
        Adj = np.vstack((np.reshape(G1[0:batch_size,batch_size:],(batch_size,10*batch_size)),tmpp[batch_size:,batch_size:]))
        Adj = np.hstack((np.reshape(G1,(batch_size+10*batch_size,batch_size)),Adj)) # [0:100,:]
        
        Adj = Adj+np.eye(Adj.shape[0])
        d = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0)
        Adj = Adj.dot(d).transpose().dot(d)
        
        

        i_c=i_c+batch_size
        model_Semi.fit([np.array(input1),np.array(input2), Adj],np.array(output1),sample_weight=np.concatenate((np.zeros((batch_size,)),np.ones((10*batch_size,)))),
                 batch_size=10*batch_size+batch_size, epochs=20, shuffle=False)
        
        if flag==1:
           yield [np.array(input1),np.array(input2), Adj], np.array(output1), np.concatenate((np.zeros((batch_size,)),np.ones((10*batch_size,))))
        else:
           yield [np.array(input1),np.array(input2), Adj], output1_[0:batch_size]
        

def generate_data_val(batch_size_test, batch_size,flag):
  i_c=0
  while True:
        input1 = []
        input2 = []

        output1 = []
        batch_counter=0
        
        if i_c>=num_val_samples:
          i_c=0

        in_drug=val_drugs[i_c:i_c+batch_size_test]
        in_prots=val_prots[i_c:i_c+batch_size_test]

        idx = np.concatenate((range(0,i_c),range(i_c+1+batch_size_test,train_drugs.shape[0])))
        idx=idx.astype(int)
        
        pindex = np.random.choice(idx.shape[0], 10*batch_size, replace=False)  
        #print(pindex)

        input1 = np.concatenate((in_drug,train_drugs[idx[pindex]]), axis=0)
        input2 = np.concatenate((in_prots,train_prots[idx[pindex]]), axis=0)


        output1 = np.concatenate((val_Y[i_c:i_c+batch_size_test],train_Y[idx[pindex]]), axis=0)
        output1 = np.reshape(output1,(batch_size+10*batch_size,1))

        G = model2.predict([np.array(input1[0:batch_size]),np.array(input2[0:batch_size])],batch_size=batch_size)
        
        output1_ = (output1>threshold)
        output1_ = output1_.astype(int)
        tmpp=np.matmul(output1_,np.transpose(output1_))+np.matmul(1-output1_,np.transpose(1-output1_))
        
        
        G1=np.array(G)
        G1 = (G1>threshold)
        G1 = G1.astype(int)
        
        G1 = np.matmul(G1,np.transpose(output1_))+np.matmul(1-G1,np.transpose(1-output1_))
        
        Adj = np.vstack((np.reshape(G1[0:batch_size,batch_size:],(batch_size,10*batch_size)),tmpp[batch_size:,batch_size:]))
        Adj = np.hstack((np.reshape(G1,(10*batch_size+batch_size,batch_size)),Adj)) # [0:100,:]
        
        Adj = Adj+np.eye(Adj.shape[0])
        d = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0)
        Adj = Adj.dot(d).transpose().dot(d)
        
        

        i_c=i_c+batch_size_test
        """for kk in range(0,10):
            model_Semi.fit([np.array(input1),np.array(input2), Adj],np.array(output1),sample_weight=np.concatenate((np.zeros((batch_size1,)),np.ones((10*batch_size1,)))),
                 batch_size=1100, epochs=1, shuffle=False)
        """
        if flag==1:
           yield [np.array(input1),np.array(input2), Adj], np.array(output1), np.concatenate((np.zeros((batch_size,)),np.ones((10*batch_size,))))
        else:
           yield [np.array(input1),np.array(input2), Adj], output1_[0:batch_size]

for epoch in range(1, NB_EPOCH+1):
    #batch_size1=100
    model_Adj.trainable=True
    model_Feat.trainable=True
    model2.fit(([np.array(train_drugs),np.array(train_prots) ]), np.array(train_Y), batch_size=512, epochs=100, 
                             validation_data=( ([np.array(val_drugs), np.array(val_prots) ]), np.array(val_Y)),  shuffle=False )
    
    #model_Adj.trainable=False
    model_Feat.trainable=False
    model_Semi.fit_generator(generator=generate_data(batch_size,1),
              steps_per_epoch = num_train_samples/batch_size, epochs=num_epoch_GCN, shuffle=False, verbose=1)#,validation_data=generate_data_val(batch_size1), validation_steps=50)
			  
			  
			  
			  
			  
# Test Phase
i_c=0
while True:
    input1 = []
    input2 = []
    output1 = []
    batch_counter=0
        
    if i_c>=num_val_samples:
        break

    in_drug=val_drugs[i_c:i_c+batch_size_test]
    in_prots=val_prots[i_c:i_c+batch_size_test]
    idx = np.concatenate((range(0,i_c),range(i_c+1+batch_size_test,train_drugs.shape[0])))
    idx=idx.astype(int)
        
    pindex = np.random.choice(idx.shape[0], num_neighbours_test, replace=False)  
    #print(pindex)

    input1 = np.concatenate((in_drug,train_drugs[idx[pindex]]), axis=0)
    input2 = np.concatenate((in_prots,train_prots[idx[pindex]]), axis=0)


    output1 = np.concatenate((val_Y[i_c:i_c+batch_size_test],train_Y[idx[pindex]]), axis=0)
    output1 = np.reshape(output1,(num_neighbours_test+batch_size_test,1))

    G = model2.predict([np.array(input1[0:batch_size_test]),np.array(input2[0:batch_size_test])],batch_size=batch_size_test)
        
    output1_ = (output1>threshold)
    output1_ = output1_.astype(int)
    tmpp=np.matmul(output1_,np.transpose(output1_))+np.matmul(1-output1_,np.transpose(1-output1_))
       
        
    G1=np.array(G)
    G1 = (G1>threshold)
    G1 = G1.astype(int)
        
    G1 = np.matmul(G1,np.transpose(output1_))+np.matmul(1-G1,np.transpose(1-output1_))
        
    Adj = np.vstack((np.reshape(G1[0:batch_size_test,batch_size_test:],(batch_size_test,num_neighbours_test)),tmpp[batch_size_test:,batch_size_test:]))
    Adj = np.hstack((np.reshape(G1,(num_neighbours_test+batch_size_test,batch_size_test)),Adj)) # [0:100,:]
        
    Adj = Adj+np.eye(Adj.shape[0])
    d = np.diag(np.power(np.array(Adj.sum(1)), -0.5).flatten(), 0)
    Adj = Adj.dot(d).transpose().dot(d)
        
        

    i_c=i_c+batch_size_test
    model_Semi.fit([np.array(input1),np.array(input2), Adj],np.array(output1),sample_weight=np.concatenate((np.zeros((batch_size_test,)),np.ones((num_neighbours_test,)))),
            batch_size=num_neighbours_test+batch_size_test, epochs=num_epoch_GCN_inference, shuffle=False)
    
    preds.append(model_Semi.predict([np.array(input1),np.array(input2), Adj],np.array(output1))[0])
print(preds)

preds_m=preds
print(preds_m)
m=keras.metrics.MeanSquaredError()
m.update_state(preds_m,val_Y)
print("MSE= {:.4f}".format(m.result().numpy()))
print(cindex_score(val_Y,preds_m))

