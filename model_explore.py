#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 10:43:13 2018

@author: cgyy2
"""


from __future__ import print_function
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
print(__doc__)

import sys
src_path = "/dsaa/shared/users/wei_liu/work/llo/src/"
sys.path.append(src_path)

import datetime
today = datetime.datetime.now().strftime("%m-%d-%Y")
fig_path = "/dsaa/shared/users/wei_liu/work/llo/result/fig/" + today +'/'
data_path =  "/dsaa/shared/users/wei_liu/work/llo/data/"



import numpy as np
import pandas as pd
import scipy.signal as ss
import colorsys
import scipy.signal as ss
from collections import OrderedDict
import math
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import time
import os

from helper import create_dir
from helper import plot_curve
from helper import my_pct
from helper import kmeans
from helper import kmeans_nd
from helper import normalize_cols
from helper import standardize_cols
from helper import list_files
from sklearn import preprocessing

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model, datasets
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


#import keras
import tensorflow as tf
#%%
### data load

df_all = pd.read_csv(data_path + 'new_df3.csv')

df_all.head()
df_all.columns
df_all.fault_ts = pd.to_datetime(df_all.fault_ts)
#%%
df_ac = df_all[df_all.loco_model_nbr == 'ES44AC']
df_dc = df_all[df_all.loco_model_nbr == 'ES44DC']

### select sub-cols
other_cols = ['eq_nr', 'fault_ts', 'f_6062_cnt', 'f_6034_cnt']

x_cols = [u'oil_pres',
          u'oil_temp', u'rpm', u'water_press',
          u'manifold_temp', u'manifold_press', u'crankcase_press',
          u'turbo_speed', 'notch', 'gross_hp']


df_ac = df_ac[other_cols + x_cols]
df_dc = df_dc[other_cols + x_cols]
df_ac.shape #899048
df_dc.shape #3043621

### drop all the Nans
df_ac.dropna(axis=0, how='any', inplace=True)
df_ac.shape #899007

### add new col of 'time diff'
for loco, mdf in df_ac.groupby('eq_nr'):    
    df_ac.loc[mdf.index, 'time_diff'] = (mdf['fault_ts']-mdf['fault_ts'].shift()).fillna(0)

df_ac.time_diff = df_ac.time_diff/pd.Timedelta('1 min')
df_ac.head()

def pick_recent_time(df, t):
    df['selected'] = False
    for loco, mdf in df.groupby('eq_nr'):
        latest_t = mdf.fault_ts.max()
        start_t = latest_t - pd.Timedelta(t)
        mdf = mdf[mdf.fault_ts >= start_t]
        df.loc[mdf.index, 'selected'] = True

pick_recent_time(df_ac,'365 days') #select the last 6 month
df_ac = df_ac[df_ac.selected==True]

pick_recent_time(df_dc,'365 days')
df_dc = df_dc[df_dc.selected==True]

df_ac = df_ac[other_cols + x_cols]
df_dc = df_dc[other_cols + x_cols]

print(df_ac.shape) #312026
print(df_dc.shape) #797368

pick_recent_time(df_ac,'90 days') #select the last 3 month for testing
test_df = df_ac[df_ac.selected==True]
train_df = df_ac[df_ac.selected==False]

test_df = test_df[other_cols + x_cols]
train_df = train_df[other_cols + x_cols]

test_df.shape #80182
train_df.shape #818866
train_df.head()
#%%
### get sample data
df_ac.head()
## randomly pick 80% of locos_has_fault, 80% of locos_no_fault to get the training set
def select_sample_data(df, seed = 1, ratio = .8):
    np.random.seed(seed)
    loco_fault_dic = {}
    for loco, mdf in df.groupby('eq_nr'):
        loco_fault_dic[loco] = mdf.f_6062_cnt.sum()
  
    fault_locos = [k for k,v in loco_fault_dic.items() if v>0]

    train_fault = np.random.choice(fault_locos, int(ratio*len(fault_locos)), replace=False)
    test_fault = set(fault_locos) - set(train_fault)
       
    normal_locos = list(set(loco_fault_dic.keys()) - set(fault_locos))
    train_normal = np.random.choice(normal_locos, int(ratio*len(normal_locos)), replace=False)
    test_normal = set(normal_locos) - set(train_normal)
    
    print (len(train_fault), len(train_normal), len(test_fault), len(test_normal))
    
    train_locos = set(train_fault).union(train_normal)
    test_locos = set(test_fault).union(test_normal)
        
    assert len( train_locos.intersection(test_locos) )== 0
   
    return train_locos, test_locos

train_locos, test_locos = select_sample_data(df_ac)

train_df = df_ac.groupby('eq_nr').filter(lambda g:g.iloc[0]['eq_nr'] in train_locos)
train_df.shape #710977

test_df = df_ac.groupby('eq_nr').filter(lambda g:g.iloc[0]['eq_nr'] in test_locos)
test_df.shape #188030

train_df.f_6062_cnt.sum() ##111
test_df.f_6062_cnt.sum() ##32


#%%  interpolate by 5mins
'''
df_ac2.set_index(['fault_ts'], inplace=True)
df_dc2.head()

dfs = []
for loco, mdf in df_ac2.groupby('eq_nr'):
#    dfs.append(mdf.resample('5min').ffill(limit=1).interpolate())
    oidx = mdf.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq='5min')
    res = mdf.reindex(oidx.union(nidx)).interpolate('index').reindex(nidx)
    dfs.append(res)
    

df_ac3 = pd.concat(dfs)

df_ac3['f62'] = 0
df_ac3.head()
df_ac2.head()

df_ac3.reset_index(inplace=True)
df_ac3.rename(columns={'index':'fault_ts'}, inplace=True)
df_ac3.fault_ts = pd.to_datetime(df_ac3.fault_ts)

for loco, mdf in df_ac2.groupby('eq_nr'):
    mdf = mdf[(mdf.f_6062_cnt == 1) | (mdf.f_6062_cnt == 2)]
    for i in mdf.index:
        print(mdf.loc[i, 'f_6062_cnt'])
        print(np.argmin(abs((df_ac3[df_ac3.eq_nr==loco].fault_ts - pd.Timestamp(i)))))
        df_ac3.at[ np.argmin(abs((df_ac3[df_ac3.eq_nr==loco].fault_ts - pd.Timestamp(i)))), 'f62'] = mdf.loc[i, 'f_6062_cnt']
        
df_ac3.f62.sum()

df_dc2.set_index(['fault_ts'], inplace=True)
dfs = []
for loco, mdf in df_dc2.groupby('eq_nr'):
#    dfs.append(mdf.resample('5min').ffill(limit=1).interpolate())
    oidx = mdf.index
    nidx = pd.date_range(oidx.min(), oidx.max(), freq='5min')
    res = mdf.reindex(oidx.union(nidx)).interpolate('index').reindex(nidx)
    dfs.append(res)
df_dc3 = pd.concat(dfs)

df_dc3.head()
df_dc3.dropna(axis=0, how='any', inplace=True)
df_dc3.reset_index(inplace=True)
df_dc3.rename(columns={'index':'fault_ts'}, inplace=True)
df_dc3.fault_ts = pd.to_datetime(df_ac3.fault_ts)
df_dc3['f62'] = 0
df_dc3.head()

df_dc3.eq_nr.unique()
for loco, mdf in df_dc2.groupby('eq_nr'):
    mdf = mdf[(mdf.f_6062_cnt == 1) | (mdf.f_6062_cnt == 2)]
    for i in mdf.index:
        print(mdf.loc[i, 'f_6062_cnt'])
        print(np.argmin(abs((df_dc3[df_dc3.eq_nr==loco].fault_ts - pd.Timestamp(i)))))
        df_dc3.at[ np.argmin(abs((df_dc3[df_dc3.eq_nr==loco].fault_ts - pd.Timestamp(i)))), 'f62'] = mdf.loc[i, 'f_6062_cnt']
        
df_dc3.f62.unique()
### ---------------------------------------------------------------
'''
#%%

### mark the prev 48 hours
def mark_time_before_event(df, mt): 
    df[mt] = np.datetime64('NaT')
    def foo(t, faults):
        for ft in faults:
            if pd.Timedelta('0h') <= (ft-t) < pd.Timedelta(mt):
                return ft
        return np.datetime64('NaT')


    for loco, mdf in df.groupby('eq_nr'):
        if mdf.f_6062_cnt.sum() == 0:
            continue
        f_6062 = mdf[(mdf.f_6062_cnt == 1) | (mdf.f_6062_cnt == 2)]['fault_ts']
        df.loc[mdf.index, mt] = mdf['fault_ts'].apply(lambda x: foo(x, f_6062))
    #new_col = 'b_' + mt
    #df[new_col]= df[mt].apply(lambda x: x is not pd.NaT)

mark_time_before_event(df_ac2, '48h')
mark_time_before_event(df_dc2, '48h')

mark_time_before_event(test_df, '48h')
mark_time_before_event(train_df, '48h')


df_ac2[(df_ac2.f_6062_cnt == 1) | (df_ac2.f_6062_cnt == 2)].shape ##37
df_dc2[(df_dc2.f_6062_cnt == 1) | (df_dc2.f_6062_cnt == 2)].shape ##62

df_ac2.head()

### cross entrophy
def cross_entropy(n):    
    arr = np.cumsum([1.0]*n)
    arr = np.exp(arr)
    return arr/sum(arr)
df_ac2['48h_score'] = .0
df_dc2['48h_score'] = .0

test_df['48h_score'] = .0
train_df['48h_score'] = .0


for (loco, t), mdf in df_ac2.groupby(['eq_nr', '48h']):
    df_ac2.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0])
    
for (loco, t), mdf in df_dc2.groupby(['eq_nr', '48h']):
    df_dc2.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0])
     
    
for (loco, t), mdf in test_df.groupby(['eq_nr', '48h']):
    test_df.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0])
    
for (loco, t), mdf in train_df.groupby(['eq_nr', '48h']):
    train_df.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0]) 

train_df['48h_score'].sum() #111
test_df['48h_score'].sum() #32


#%%
### validate through plots    
create_dir(fig_path + '/locos')
i = 0
for loco, mdf in test_df.groupby('eq_nr'):
    if mdf.f_6062_cnt.sum() == 0:
        continue
    if i > 10:
        break
    i += 1
    ts = mdf.fault_ts
    earliest_time = min(ts)    
    pdf = PdfPages(os.path.join(fig_path+'/locos/', str(loco) + '.pdf'))
    for k, frame in mdf.groupby(np.floor((ts - earliest_time).astype('timedelta64[M]')/1.0)):
        fig, ax = plt.subplots(2, figsize=(22,10), sharex=True)    
        ax[0].plot(frame.fault_ts.tolist(), frame['oil_pres'].tolist(), 'k-', 
                   marker='o',ms=1, linewidth=.1)
        ax[0].scatter(frame[frame.f_6062_cnt != 0].fault_ts.tolist(), 
                       frame[frame.f_6062_cnt != 0].oil_pres.tolist(),
                       c = 'g', s=200)
        ax[1].plot(frame.fault_ts.tolist(), frame['48h_score'].tolist(),
                   'r-',ms=5,linewidth=.1)
        pdf.savefig()
        plt.close()
    pdf.close()
'''=========================================================================================='''
#%%
from subprocess import check_output
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import  train_test_split
import time #helper libraries
from numpy import newaxis



test_df.drop(['48h'], axis=1, inplace=True)
train_df.drop(['48h'], axis=1, inplace=True)

df_ac2.dropna(axis=0, how='any', inplace=True)
df_dc2.dropna(axis=0, how='any', inplace=True)


df_ac2.shape ##67148
df_dc2.shape ##248433

test_df.shape ##51041
train_df.shape

def batch_locos(df, batch_size=100):
    tmp = []
    for loco, mdf in df.groupby('eq_nr'):
        n_batches = mdf.shape[0]//batch_size
        mdf = mdf.iloc[:n_batches * batch_size, :]
        tmp.append(mdf)
    return pd.concat(tmp)

## seperate the locos by 100
df_ac3 = batch_locos(df_ac2)
df_dc3 = batch_locos(df_dc2)

test_df = batch_locos(test_df)
train_df = batch_locos(train_df)


test_df.shape ##49600
train_df.shape


other_cols = ['eq_nr', 'fault_ts', 'f_6062_cnt', 'f_6034_cnt']


features = ['oil_pres','oil_temp', 'rpm', 'water_press',
          'manifold_temp', 'manifold_press', 'crankcase_press',
          'turbo_speed', 'notch', 'gross_hp', 'time_diff', '48h_score']

features_norm = [i +'_normed' for i in features]


scaler_ac = StandardScaler()
scaler_ac.fit(df_ac3[cols])

scaler_dc = StandardScaler()
scaler_dc.fit(df_dc3[cols])


scaler_train = StandardScaler()
scaler_train.fit(train_df[features])
train_normed = scaler_train.transform(train_df[features])
train_normed = pd.DataFrame(train_normed, columns=features_norm)
train_normed.head()

scaler_test = StandardScaler()
scaler_test.fit(test_df[features])
test_normed = scaler_test.transform(test_df[features])
test_normed = pd.DataFrame(test_normed, columns=features_norm)
test_normed.head()

train_normed.shape
test_normed.shape

train.head()
#%%

df_ac3.shape ##15400
df_dc3.shape ##62300

df_ac3.to_csv(data_path + 'df_ac3.csv')
df_dc3.to_csv(data_path + 'df_dc3.csv')

df_ac3_normed.to_csv(data_path + 'df_ac3_norm.csv')
df_dc3_normed.to_csv(data_path + 'df_dc3_norm.csv')
#%%
## build lstm rnn model
lstm_size = 256 # the number of hidden units in the LSTM cell
lstm_layers = 2 # how many layers of lstm, 2 is already good to use
#batch_size = 50 # how many sample you pass through one forward-backward iter
learning_rate = 0.0005

num_time_steps = 100

input_features = 12
output_features = 12 # num of features for one input

tf.reset_default_graph()


# Add nodes to the graph
#with tf.Graph().as_default() as graph:
    ## input size: (batch_size, time_steps, features)
inputs_ = tf.placeholder(tf.float32, [None, num_time_steps, input_features], name='inputs')
labels_ = tf.placeholder(tf.float32, [None, num_time_steps, output_features], name='targets')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) 

cells = []
for _ in range(lstm_layers):   
    cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells.append(cell)

cells = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)

outputs, state = tf.nn.dynamic_rnn(cells,
                                   inputs_,
                                   time_major=False, # default
                                   #sequence_length=[10]*batch_size,
                                   dtype=tf.float32)

predictions = tf.contrib.layers.fully_connected(outputs,
                                                num_outputs=output_features,
                                                activation_fn=None)

#loss = get_loss(get_batch(Y).reshape([batch_size, 1]), predictions)
cost = tf.losses.mean_squared_error(labels_, predictions)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
#%%
    # Your basic LSTM cell
#    lstm = tf.contrib.rnn.BasicLSTMCell(num_units = lstm_size,
#                                        activation = tf.nn.relu)
#    
#    # Add dropout to the cell
#    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
#    
#    # Stack up multiple LSTM layers, for deep learning
#    cell = tf.contrib.rnn.MultiRNNCell([drop] * lstm_layers)
#    
#    # Getting an initial state of all zeros
#    #initial_state = cell.zero_state(batch_size, tf.float32)
#
#
#
#    outputs, final_state = tf.nn.dynamic_rnn(cell=cell, inputs=inputs_, dtype=tf.float32)

#
#    predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1, 
#                                                    activation_fn=tf.sigmoid)
#    
#    cost = tf.losses.mean_squared_error(labels_, outputs)    
#    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
train_df.shape
train_normed.shape

test_df.shape
test_normed.shape

features
features_norm

train_df.reset_index(inplace=True)
train = pd.concat([train_df, train_normed], axis=1)
train.head()
train.shape

test_df.reset_index(inplace=True)
test = pd.concat([test_df, test_normed], axis=1)
test.head()
test.shape

train.to_csv(data_path + 'train_ac.csv')
test.to_csv(data_path + 'test_ac.csv')
#%%
other_cols
train.head()
def split_data_2(data, num_steps):
    infos = []
    train_x = []
    train_y = []
    for loco, mdf in data.groupby('eq_nr'):
        n_batches = mdf.shape[0]//num_steps
        for i in range(n_batches):
            info = mdf.iloc[i*num_steps : (i+1)*num_steps, :][other_cols]
            x = mdf.iloc[i*num_steps : (i+1)*num_steps, :][features_norm]            
            y = mdf.iloc[i*num_steps+1: (i+1)*num_steps+1, :][features_norm]
            if(x.shape[0] != y.shape[0]):
                break
            infos.append(info)
            train_x.append(x.values)
            train_y.append(y.values)
    return infos, train_x, train_y



split_data_2(train, 100)

#def get_batch_3(arr, b_size=5):    
#    n_locos, slice_size, n_features = arrs[0].shape
#    n_batches = int(slice_size/num_steps)
#    for b in range(n_batches):
#        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]

def get_batches_3(x, y, batch_size=10):
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]


arr = np.random.randint(15,size=(15, 4))
    
data = pd.DataFrame(arr, index=list("aaaaaaaaabbbbbb"))
data

data_x, data_y = split_data_2(data, 5)

np.array(data_x).shape
np.array(data_x).reshape(-1,4)

data_y

np.array(data_x).shape

for x,y in get_batches_3(data_x, data_y):
    print(x)    
    print(y)
    print('-------------------')
    
    
#%%
train_x, train_y = split_data_2(train, 100)

test_info, test_x, test_y = split_data_2(test, 100)

len(test_x[0])
feed_in_test_x = 


test_x = pd.DataFrame(np.array(test_x).reshape(-1, 16), columns=other_cols+features)

test_x[0]

np.array(test_x).shape
np.array(train_y).shape
train_x[0]
train_y[0]
len(train_x)
train.shape
#%%
epoches = 1000
feed_batch_size = 1000
losses = []
saver = tf.train.Saver()
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    
    iters = 0
    for e in range(epoches):
        ## train
        start = time.time()
        for x,y in get_batches_3(np.array(train_x), np.array(train_y), batch_size=feed_batch_size):
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob : .9
                    }
            loss, _ = sess.run([cost,
                               optimizer], 
                               feed_dict=feed)
            
    
            print('epoch = {}, iters = {}, mse = {}'.format(e, iters, loss))
            iters += 1
            losses.append(loss)
        print('time used = {}'.format(time.time() - start))
    saver.save(sess, "./rnn_time_series_model_3")
fig, ax = plt.subplots(figsize=(22,10)) 
ax.plot(losses, 'r-', marker='o')
fig.show()
np.array(test_x).shape

train_preds
test_y = np.array(test_y).reshape(-1, 12)
test
test_info = np.array(test_info).reshape(-1,4)
type(test_info[0])
test_info = pd.concat(test_info)
test_info.head()


test_pred = pd.DataFrame(train_preds, columns = ['_pred'+i for i in features])
test_true = pd.DataFrame(test_y, columns = ['_true'+i for i in features])
test_pred.head()

test_res = pd.concat([test_info, test_true, test_pred], axis=1)
test_res.head()


np.array(test_y).shape

test.shape
#%%  
#test_preds = []
train_preds = []
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_3")
    
    for i in range(np.array(test_x).shape[0]):
        x_new = test_x[i].reshape(-1, num_time_steps, input_features)
        #x_new = test_x[i,:,:].reshape(-1, num_time_steps, input_features)
        y_pred = sess.run(predictions, feed_dict={inputs_:x_new,
                                                  keep_prob:1.0})
        train_preds.append(y_pred)
        #test_preds.append(y_pred)
        
    #test_preds = np.stack(test_preds).reshape(-1, input_features)
    train_preds = np.stack(train_preds).reshape(-1, input_features)

#%% 
def get_batches(x, y, batch_size=100):
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        

def get_batch_2(arrs, num_steps):    
    batch_size, slice_size, _= arrs[0].shape    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]
        
def split_data(data, batch_size, num_steps, split_frac=0.9):    
    slice_size = batch_size * num_steps #1200
    
    n_batches = int(len(data) / slice_size) ##587
    
    # Drop the last few characters to make only full batches
    x = data[: n_batches*slice_size]
    y = data[1: n_batches*slice_size + 1]
    assert n_batches*slice_size + 1 <= data.shape[0]
    
    # Split the data into batch_size slices, then stack them into a 2D matrix 
    x = np.stack(np.split(x, batch_size))
    y = np.stack(np.split(y, batch_size))
    
    # Now x and y are arrays with dimensions batch_size x n_batches*num_steps
    
    # Split into training and validation sets, keep the virst split_frac batches for training
    split_idx = int(n_batches*split_frac)
    train_x, train_y= x[:, :split_idx*num_steps], y[:, :split_idx*num_steps]
    val_x, val_y = x[:, split_idx*num_steps:], y[:, split_idx*num_steps:]
    return train_x, train_y, val_x, val_y
#%%

train_x, train_y, val_x, val_y = split_data(train[features_norm], 12, 100, 1)

test_x, test_y, _, _  = split_data(test_normed, 12, 100, 1)

train_x, train_y, val_x, val_y = split_data(train_normed, 12, 100, .9)


train_x.shape #(12, 5400, 11) -> (batch_size, size_of_one_batch, feature_size)

for b, (x, y) in enumerate(get_batch_2([train_x, train_y], num_time_steps), 1):
    print(x.shape, num_time_steps)


#%%
epochs = 300
validate_every_n = 1000
#init = tf.global_variables_initializer()
saver = tf.train.Saver()
iters = []
mses = []
val_mses_mean = []
with tf.Session() as sess:   
    init = tf.global_variables_initializer()
    sess.run(init)
    n_batches = int(train_x.shape[1]/num_time_steps)
    iterations = n_batches * epochs
    for e in range(epochs):
        ## train
        for b, (x, y) in enumerate(get_batch_2([train_x, train_y], num_time_steps), 1):            
            iteration = e*n_batches + b
            start = time.time()
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob : .9
                    }
            loss, _ = sess.run([cost,
                               optimizer], 
                               feed_dict=feed)
            
            if iteration % 20 == 0:    
                mse = cost.eval(feed_dict=feed)
                print('iteration = {}, mse = {}'.format(iteration, mse))
                iters.append(iteration)
                mses.append(mse)
        ## validate every 1000?
        if (iteration % validate_every_n == 0) or (iteration == iterations):
            val_loss = []
            for b, (x, y) in enumerate(get_batch_2([val_x, val_y], num_time_steps), 1):            
                iteration = e*n_batches + b
                start = time.time()
                feed = {inputs_: x,
                        labels_: y,
                        keep_prob : 1.0
                        }
                loss, _ = sess.run([cost,
                                   optimizer], 
                                   feed_dict=feed)
                
                val_loss.append(loss)
                
            print('Validation loss: {}'.format(np.mean(val_loss)))
            val_mses_mean.append(np.mean(val_loss))
    # Save Model for Later
    saver.save(sess, "./rnn_time_series_model_2")
    

#%%  
fig, ax = plt.subplots(figsize=(22,10)) 
ax.plot( mses, 'b-', marker='o')

ax.set_title('lstm_size={}, lstm_layers={}, learning_rate={}, batch_size={},\n\
          time_steps={}, epochs={}, keep_prob={}'.format(lstm_size, lstm_layers, 
          learning_rate, 12, 100, 300, 0.9))
ax.set_xlabel('iteration #')
ax.set_ylabel('mse')
plt.tight_layout()
plt.show()
           
#%%
#preds = []
test_x.shape
train_x.shape
#test_preds = []
train_preds = []
with tf.Session() as sess:
    saver.restore(sess, "./rnn_time_series_model_2")
    
    for i in range(train_x.shape[0]):
        x_new = train_x[i,:,:].reshape(-1, num_time_steps, input_features)
        #x_new = test_x[i,:,:].reshape(-1, num_time_steps, input_features)
        y_pred = sess.run(predictions, feed_dict={inputs_:x_new,
                                                  keep_prob:1.0})
        train_preds.append(y_pred)
        #test_preds.append(y_pred)
        
    #test_preds = np.stack(test_preds).reshape(-1, input_features)
    train_preds = np.stack(train_preds).reshape(-1, input_features)
#    preds = np.stack(preds).reshape(-1, input_features)
#%%

preds_denorm = scaler_test.inverse_transform(test_preds[:-1, :])
test_df.shape
test_df_used = test_df.iloc[1:test_preds.shape[0], :]
test_df_used.reset_index(inplace=True)
assert test_df_used.shape[0] == preds_denorm.shape[0]
test_df_res = pd.concat([test_df_used, pd.DataFrame(preds_denorm, columns=new_cols)], axis=1)
test_df_res.head()

#%%

train_preds_denorm = scaler_test.inverse_transform(train_preds[:-1, :])
train_df_used = train_df.iloc[1:train_preds.shape[0], :]
train_df_used.reset_index(inplace=True)
train_preds_denorm.shape
train_df_used.shape

assert train_df_used.shape[0] == train_preds_denorm.shape[0]
train_df_res = pd.concat([train_df_used, pd.DataFrame(train_preds_denorm, columns=new_cols)], axis=1)
train_df_res.head()
#%%


df_ac3_used = df_ac3.iloc[1:preds.shape[0], :]

assert df_ac3_used.shape[0] == preds_denorm.shape[0]

df_ac3_used.reset_index(inplace=True)

new_cols = [i+'_pred' for i in cols]
df_res = pd.concat([df_ac3_used, pd.DataFrame(preds_denorm, columns=new_cols)], axis=1)
df_res.head()

df_res.to_csv(data_path + 'df_test_pred.csv')

create_dir(fig_path+'/test_4')

for loco, mdf in test_res.groupby('eq_nr'):
    ts = mdf.fault_ts
    earliest_time = min(ts)
    name = str(loco)
    if mdf['f_6062_cnt'].sum() != 0:
        name += '_fault'
    pdf = PdfPages(os.path.join(fig_path+'/test_4/', name + '.pdf'))
    for k, frame in mdf.groupby(np.floor((ts - earliest_time).astype('timedelta64[M]')/1.0)):
        fig, ax = plt.subplots(3, figsize=(22,10), sharex=True)    
        ax[0].plot(frame.fault_ts.tolist(), frame['_trueoil_pres'].tolist(), 'k-', 
                   marker='o',ms=5, linewidth=1, label='real_data')
        ax[0].plot(frame.fault_ts.tolist(), frame['_predoil_pres'].tolist(), 'g-', 
                   marker='o',ms=5, linewidth=1, label='pred')
        
            
        for xi in frame[frame.f_6034_cnt != 0].fault_ts.tolist():
            ax[0].axvline(xi, color='magenta', linewidth=3, label='6034')
        
        for xi in frame[frame.f_6062_cnt != 0].fault_ts.tolist():
            ax[0].axvline(xi, color='r', linewidth = 3, label='6062')
        ax[0].legend()
        
        ax[1].plot(frame.fault_ts.tolist(), frame['_trueoil_temp'].tolist(), 'k-', 
                   marker='o',ms=5, linewidth=1)
        ax[1].plot(frame.fault_ts.tolist(), frame['_predoil_temp'].tolist(), 'g-', 
                   marker='o',ms=5, linewidth=1)
        
        #frame.set_index(['fault_ts'], inplace=True)
        roll1 = frame['_true48h_score']#.rolling('4h').sum()
        roll2 = frame['_pred48h_score']#.rolling('4h').sum()
        ax[2].plot(frame.fault_ts.tolist(), roll1.tolist(), 'b-', marker = 'o', label='true')
        ax[2].plot(frame.fault_ts.tolist(), roll2.tolist(), 'r-', marker = 'o', label ='pred')
        ax[2].legend()
        ax[2].set_ylim(0, 100)
        plt.tight_layout()
        pdf.savefig()
        plt.close()
    pdf.close()
    
df_ac3.eq_nr.nunique()
df_ac.eq_nr.nunique()
print(22/171.0)
#%%
train_x.shape
preds[:2,:]
df_ac3.shape

data = [[0, 0], [0, 0], [1, 1], [1, 1]]

preprocessing.scale(data)


preds_denorm = scaler.inverse_transform(preds[:-1, :10])
preds_denorm = np.c_[preds_denorm, preds[:, 10]]

origin = scaler.inverse_transform(df_ac3.iloc[1:15000, 3:-2])

origin.shape
preds_denorm.shape

origin[:2, :]
preds_denorm[:2,:]


df_ac3.iloc[1:2, 3:-2]



df_ac3.iloc[:2,:]


for loco, mdf in df_ac3[1:15000, :].groupby('eq_nr'):
    
    



df_ac3.head()
train_x.shape



x_new.shape #(15, 100, 11)
y_pred.shape #(15, 100, 11)
preds = np.stack((y.reshape(-1, 2), y.reshape(-1, 2))).reshape(-1,2)
preds

train_x.shape


x_new[0, :, :10]

df_ac3.shape

ts = df_ac3[:1500].fault_ts.reshape(13, 100, 1)
fs = df_ac3[:1500].f_6062_cnt.reshape(13, 100, 1)


ts.shape
fs[12,:,0]


for i in range(x_new.shape[0]):
    pres = scaler.inverse_transform(x_new[i, :, :10])[1:,0] 
    pres_ = scaler.inverse_transform(y_pred[i, :, :10])[:-1,0]
    
    x_ = ts[i, :-1, 0]
    
    temp = scaler.inverse_transform(x_new[i, :, :10])[1:,1] 
    temp_ = scaler.inverse_transform(y_pred[i, :, :10])[:-1,1] 
    fig, (ax1,ax2, ax3) = plt.subplots(3, figsize=(22,10))
    
    ax1.plot(x_, pres, 'r-', marker='o')
    ax1.plot(x_, pres_, 'g-', marker='o')  
       
    ax2.plot(x_, temp, 'r-', marker='o')
    ax2.plot(x_, temp_, 'g-', marker='o')
    
    ax3.plot(x_, x_new[0, 1:, 10], 'r-', marker='o')
    ax3.plot(x_, y_pred[0, :-1, 10], 'g-', marker='o')
    #ax3.axvline(x_[fs[i,:,0]!=0])  
    
    
    
    
    






#%%
print("Train set: \t\t{}".format(train_x.shape))

epochs = 10

with graph.as_default():
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    iteration = 1
    for e in range(epochs):
        state = sess.run(initial_state)
        
        for ii, (x, y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
            feed = {inputs_: x,
                    labels_: y[:, None],
                    keep_prob: 0.5,
                    initial_state: state}
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)
            
            if iteration%5==0:
                print("Epoch: {}/{}".format(e, epochs),
                      "Iteration: {}".format(iteration),
                      "Train loss: {:.3f}".format(loss))

            if iteration%25==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_x, val_y, batch_size):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keep_prob: 1,
                            initial_state: val_state}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                print("Val acc: {:.3f}".format(np.mean(val_acc)))
            iteration +=1
    saver.save(sess, "checkpoints/sentiment.ckpt")






 
    #%%

def get_batches(x, y, batch_size=100):    
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        
        
    
#%%
### create new columns
def mark_time_before_event(df, mt, prev_t, post_t): 
    df[mt] = np.datetime64('NaT')
    def foo(t, faults):
        for ft in faults:
            if pd.Timedelta('0h') < (ft-t) < pd.Timedelta(mt):
                return ft
        return np.datetime64('NaT')
    df['fault_ts'] = pd.to_datetime(df['fault_ts'])

    for loco, mdf in df.groupby('eq_nr'):
        if mdf.f_6062_cnt.sum() == 0:
            continue
        f_6062 = mdf[mdf.f_6062_cnt != 0]['fault_ts']
        df.loc[mdf.index, mt] = mdf['fault_ts'].apply(lambda x: foo(x, f_6062))
    new_col = 'b_' + mt
    df[new_col]= df[mt].apply(lambda x: x is not pd.NaT)
    df['label_'+mt] = df[new_col].apply(lambda x: {True:1, False:0}[x])
mark_time_before_event(df, '24h')
df.head()
df.label_24h.sum()
df.label_24h.unique()
#%%
from sklearn.preprocessing import StandardScaler


other_cols = ['eq_nr', 'loco_model_nbr', 'fault_ts', 'f_6062_cnt']
x_cols = [u'oil_pres',
          u'oil_temp', u'rpm', u'water_press',
          u'manifold_temp', u'manifold_press', u'crankcase_press',
          u'turbo_speed', 'notch', 'gross_hp']

df_ac = df_ac[other_cols + x_cols]

def scale_df(df):   
    scaler = StandardScaler()
    scaler.fit(df[x_cols])
    df[x_cols] = scaler.transform(df[x_cols])

df_ac.dropna(axis = 0, how = 'any', inplace=True)

scale_df(df_dc)

df_dc.to_csv(data_path + 'df_dc.csv')
df_ac.head()

df_ac.shape
df_ac.drop_duplicates(subset=['eq_nr', 'fault_ts'],inplace=True)



### pick days before and after a 6062 event
def pick_recent_time(df, col_name, prev_t, post_t):
    df[col_name] = False
    def foo(t, markers): # t is the marker-time        
        for x in markers:
            if pd.Timedelta('-'+prev_t) < (x-t) < pd.Timedelta(post_t):
                return True
        return False
    for loco, mdf in df.groupby('eq_nr'):
        if mdf.f_6062_cnt.sum() == 0:
            continue
        f_6062 = mdf[mdf.f_6062_cnt != 0]['fault_ts']
        df.loc[mdf.index, col_name] = mdf['fault_ts'].apply(lambda x: foo(x, f_6062)) 

pick_recent_time(df_ac, "ten_five_ds", '10d', '5d')



pd.Timedelta('-10d')  

df = pd.DataFrame({'ts':pd.date_range(start='2018-02-23 16:06:00',periods=8,freq='20min'),
                   'v':np.random.normal(1,1,8)})
df.set_index(['ts'], inplace=True)
df

df.resample('4min').ffill(limit=1)

.ffill(limit=1).interpolate()

df.ts.interpolate('1min')
np.interp(pd.date_range(start='2015-01-02 09:04:00',periods=8,freq='1min'), df.ts, df.v)
df

#%%
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

#%%
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, shape=[None, None], name='input')
    targets = tf.placeholder(tf.int32, shape=[None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, shape=None, name='learning_rate')

    return inputs, targets, learning_rate

def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size, state_is_tuple=True)
    num_layers = 2
    cell = tf.contrib.rnn.MultiRNNCell([lstm] * num_layers)
    initial_state = cell.zero_state(batch_size, tf.float32)
    initial_state = tf.identity(initial_state, name='initial_state')
    return cell, initial_state




#%%

x_cols = [u'oil_pres',
          u'oil_temp', u'rpm', u'water_press',
          u'manifold_temp', u'manifold_press', u'crankcase_press', u'turbo_speed']
RANDOM_STATE = 42
FIG_SIZE = (22, 10)


models_dict = {}
logreg = linear_model.LogisticRegression(penalty = 'l2', C=1e2, solver= 'liblinear')
svm = SVC(C = 1e5,kernel="rbf", gamma = 'auto')
nn =  MLPClassifier(activation='relu', solver='lbfgs', alpha=1e-7, hidden_layer_sizes=(128, 99), random_state=1)
rf = RandomForestClassifier(n_estimators=300, bootstrap=True, criterion='gini', max_depth=None, random_state=0)
for loco_model, mdf in df_all.groupby('loco_model_nbr'):
    if loco_model not in ['ES44DC', 'ES44AC']:
        continue
    print(loco_model)
    mdf.dropna(subset=x_cols, how='any', axis=0, inplace=True)

    print(mdf.label_24h.sum())    
    features, target = mdf[x_cols].values, mdf['label_24h'].values

    # Make a train/test split using 30% test size
    
#    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
#    
#    train_index, test_index = sss.split(features, target).next()
#    
#    X_train, X_test = features[train_index], features[test_index]
#    y_train, y_test = target[train_index], target[test_index]
    
    #all_X = np.r_[X_train, X_test]
    #idx = np.r_[train_index, test_index]
    
    
    
    X_train, X_test, y_train, y_test = train_test_split(features, target,
                                                        test_size=0.2,
                                                        random_state=RANDOM_STATE,                                                       
                                                        stratify=target)
    
    
    
    # Fit to data and predict using pipelined scaling, GNB and PCA.
    std_clf = make_pipeline(StandardScaler(),  rf) #PCA(n_components=3),
    std_clf.fit(X_train, y_train)
    
    print ('save to dict')
    models_dict[loco_model] = std_clf
    
#    cv_score = cross_val_score(std_clf, X_train, y_train, cv=5)  
#    cv_score = [round(i*100, 2) for i in cv_score]
#    print (cv_score, 'cv_score')
    
    pred_test_std = std_clf.predict(X_test)

    ac_score = metrics.accuracy_score(y_test, pred_test_std)
    
    ac_score = round(ac_score*100, 2)
    cnf_matrix = confusion_matrix(y_test, pred_test_std)
    
    
    
    print(cnf_matrix, 'confusion matrix')
    
    #df_all.loc[mdf.index, 'pred'] = std_clf.predict(mdf[x_cols].values)
    
#    # Extract PCA from pipeline
#    pca_std = std_clf.named_steps['pca']
#    
#    # Scale and use PCA on X_train data for visualization.
#    scaler = std_clf.named_steps['standardscaler']
#    X_train_std = pca_std.transform(scaler.transform(X_train))
#    
#    # visualize standardized vs. untouched dataset with PCA performed
#    fig, ax = plt.subplots(figsize=FIG_SIZE)
#    labels = ['normal', 'anormaly']
#    
#    for l, c, m in zip((0, 1), ('blue', 'red'), ('o', 'o')):
#        ax.scatter(X_train_std[y_train == l, 0], X_train_std[y_train == l, 1],
#                    color=c,
#                    label=labels[l],
#                    marker=m,
#                    edgecolor='k'
#                    )
#    
#    ax.set_title('Standardized training dataset after PCA, loco_model={0}, \
#                 cv_score={1}%, acc_on_test={2}%'.format(loco_model,cv_score, ac_score))
#    ax.set_xlabel('1st principal component')
#    ax.set_ylabel('2nd principal component')
#    ax.legend(loc='upper right')
#    ax.grid()
#    plt.tight_layout()
#    plt.show()
    
#%%
df.head()
for loco_model, mdf in df_all.groupby('loco_model_nbr'):
    if loco_model not in ['ES44DC', 'ES44AC']:
        continue
    clf = models_dict[loco_model]
    mdf.dropna(subset=x_cols, how='any', axis=0, inplace=True)
    print(mdf[x_cols].values[:10])
    print(clf.predict(mdf[x_cols].values).sum())
    df_all.loc[mdf.index, 'pred'] = clf.predict(mdf[x_cols].values)
    
df.pred.sum()

dfa = pd.DataFrame({'a':[1,2,3,4,5,6,7]})

def foo(a):
    print((a))
    return np.mean(a)
dfa.rolling(window=3,center=True).apply(eval('np.mean'))


df.shape

df_all.shape

df.head()

df.reset_index(inplace=True, drop=True)
df.index = pd.to_datetime(df.index)
df.set_index(['fault_ts'], inplace=True)

ts = [1,2,4]
unit = 'h'
fs = ['np.mean', 'np.std']
def add_feature(df):
    prefix='rolling_'
    for t in ts:
        for f in fs:
            df[prefix+str(t)+unit+f[f.find('.'):]] = df.rolling(window=str(t)+unit).apply(eval(f))


df.sort_values(['eq_nr', 'fault_ts'], inplace=True)



df['oil_temp-mean_4h'] = 0


for loco, mdf in df.groupby('eq_nr'):
    print (df.loc[mdf.index, 'oil_temp-mean_4h'].shape),   
    mdf.set_index(['fault_ts'], inplace=True)
    (mdf['oil_temp'].rolling('4h').mean()
    


       
    prefix='rolling_'
    for t in ts:
        for f in fs:
            df.loc[mdf.index, prefix+str(t)+unit+f[f.find('.'):]] = df.rolling(window=str(t)+unit).apply(eval(f))
    


