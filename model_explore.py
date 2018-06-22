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


import keras
import tensorflow
#%%
### data load

df_all = pd.read_csv(data_path + 'new_df3.csv')

df_all.head()
df_all.columns
df_all.fault_ts = pd.to_datetime(df_all.fault_ts)
#%%
df_ac = df_all[df_all.loco_model_nbr == 'ES44AC']
df_dc = df_all[df_all.loco_model_nbr == 'ES44DC']


other_cols = ['eq_nr', 'fault_ts', 'f_6062_cnt']
x_cols = [u'oil_pres',
          u'oil_temp', u'rpm', u'water_press',
          u'manifold_temp', u'manifold_press', u'crankcase_press',
          u'turbo_speed', 'notch', 'gross_hp']

df_ac = df_ac[other_cols + x_cols]
df_dc = df_dc[other_cols + x_cols]
df_ac.shape #899048
df_dc.shape #3043621

def pick_recent_time(df, t):
    df['selected'] = False
    for loco, mdf in df.groupby('eq_nr'):
        latest_t = mdf.fault_ts.max()
        start_t = latest_t - pd.Timedelta(t)
        mdf = mdf[mdf.fault_ts >= start_t]
        df.loc[mdf.index, 'selected'] = True

pick_recent_time(df_ac,'180 days') #select the last 4 month
df_ac = df_ac[df_ac.selected==True]

pick_recent_time(df_dc,'180 days')
df_dc = df_dc[df_dc.selected==True]

df_ac = df_ac[other_cols + x_cols]
df_dc = df_dc[other_cols + x_cols]

print(df_ac.shape) #149090
print(df_dc.shape) #333431

#%%
### get sample data
df_ac.head()

def select_sample_data(df):
    loco_fault_dic = {}
    for loco, mdf in df.groupby('eq_nr'):
        loco_fault_dic[loco] = mdf.f_6062_cnt.sum()
    loco_fault_dic
    
    fault_locos = [k for k,v in loco_fault_dic.items() if v>0]
    fault_locos
    
    no_fault_locos = np.random.choice(list(set(loco_fault_dic.keys()) - set(fault_locos)),
                                      len(fault_locos))
    no_fault_locos
    
    selected = set(no_fault_locos).union( set(fault_locos))
    return selected

selected = select_sample_data(df_ac)
print(selected)
df_ac2 = df_ac.groupby('eq_nr').filter(lambda g:g.iloc[0]['eq_nr'] in selected)
df_ac2.shape

selected = select_sample_data(df_dc)
print(selected)
df_dc2 = df_dc.groupby('eq_nr').filter(lambda g:g.iloc[0]['eq_nr'] in selected)
df_dc2.shape


print(df_ac2.shape) #16481
print(df_dc2.shape) #64531

loco_fault_dic={}
for loco, mdf in df_ac2.groupby('eq_nr'):    
    loco_fault_dic[loco] = mdf.f_6062_cnt.sum()
loco_fault_dic

loco_fault_dic={}
for loco, mdf in df_dc2.groupby('eq_nr'):    
    loco_fault_dic[loco] = mdf.f_6062_cnt.sum()
loco_fault_dic


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

df_ac2[(df_ac2.f_6062_cnt == 1) | (df_ac2.f_6062_cnt == 2)].shape ##16
df_dc2[(df_dc2.f_6062_cnt == 1) | (df_dc2.f_6062_cnt == 2)].shape ##34


### cross entrophy
def cross_entropy(n):    
    arr = np.cumsum([1.0]*n)
    arr = np.exp(arr)
    return arr/sum(arr)
df_ac2['48h_score'] = .0
df_dc2['48h_score'] = .0

for (loco, t), mdf in df_ac2.groupby(['eq_nr', '48h']):
    df_ac2.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0])
    
for (loco, t), mdf in df_dc2.groupby(['eq_nr', '48h']):
    df_dc2.loc[mdf.index, '48h_score'] = cross_entropy(mdf.shape[0])
     
df_ac2.head()
#%%
### validate through plots    
create_dir(fig_path + '/locos')
i = 0
for loco, mdf in df_ac2.groupby('eq_nr'):
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


df_ac3.head()

other_cols = ['eq_nr', 'fault_ts', 'f_6062_cnt']

x_cols = [u'oil_pres',
          u'oil_temp', u'rpm', u'water_press',
          u'manifold_temp', u'manifold_press', u'crankcase_press',
          u'turbo_speed', 'notch', 'gross_hp']
y_cols = ['48h_score']

def scale_df(df):   
    scaler = StandardScaler()
    scaler.fit(df[x_cols])
    df[x_cols] = scaler.transform(df[x_cols])

scale_df(df_dc2)
df_ac2.shape
df_ac2.head()

#%%
def batch_locos(df, batch_size=100):
    batch_size = 100
    tmp = []
    for loco, mdf in df.groupby('eq_nr'):
        n_batches = mdf.shape[0]//batch_size
        mdf = mdf.iloc[:n_batches * batch_size, :]
        tmp.append(mdf)
    return pd.concat(tmp)
        

df_ac3 = batch_locos(df_ac2)
df_dc3 = batch_locos(df_dc2)

df_ac3.shape ##15400
df_dc3.shape ##62300

df_ac3.to_csv(data_path + 'df_ac3.csv')
df_dc3.to_csv(data_path + 'df_dc3.csv')
#%%
## build lstm rnn model
lstm_size = 256 # the number of hidden units in the LSTM cell
lstm_layers = 2 # how many layers of lstm, 2 is already good to use
batch_size = 50 # how many sample you pass through one forward-backward iter
learning_rate = 0.001

num_time_steps = 100

input_features = 11
output_features = 11 # num of features for one input

tf.reset_default_graph()


# Add nodes to the graph
with tf.Graph().as_default() as graph:
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
    
    
    
def get_batches(x, y, batch_size=100):
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]
        

def  get_batch_2(arrs, num_steps):
    
    batch_size, slice_size, _= arrs[0].shape
    
    n_batches = int(slice_size/num_steps)
    for b in range(n_batches):
        yield [x[:, b*num_steps: (b+1)*num_steps] for x in arrs]
        
def split_data(data, batch_size, num_steps, split_frac=0.9):
   
    slice_size = batch_size * num_steps
    n_batches = int(len(data) / slice_size)
    
    # Drop the last few characters to make only full batches
    x = data[: n_batches*slice_size]
    y = data[1: n_batches*slice_size + 1]
    
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
df_ac3.shape
all_cols = x_cols + y_cols


train_x, train_y, val_x, val_y = split_data(df_ac3[all_cols], 10, 50)
train_x.shape #(10, 1350, 11) -> (batch_size, size_of_one_batch, feature_size)

for b, (x, y) in enumerate(get_batch_2([train_x, train_y], num_time_steps), 1):
    print(x.shape, num_time_steps)


#%%
epochs = 30

#init = tf.global_variables_initializer()
#saver = tf.train.Saver(max_to_keep=100)
with tf.Session(graph=graph) as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    n_batches = int(train_x.shape[1]/num_time_steps)
    iterations = n_batches * epochs
    for e in range(epochs):

        for b, (x, y) in enumerate(get_batch_2([train_x, train_y], num_time_steps), 1):            
            iteration = e*n_batches + b
            start = time.time()
            feed = {inputs_: x,
                    labels_: y,
                    keep_prob : .9
                    }
            sess.run(optimizer, feed_dict=feed)
            
            if iteration % 20 == 0:    
                mse = cost.eval(feed_dict=feed)
                print(iteration, "\tMSE:", mse)
    
    # Save Model for Later
    #saver.save(sess, "./rnn_time_series_model")
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
    


