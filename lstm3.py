# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


#%%
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 10:10:36 2018

@author: cgyy2
"""

import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
#%% generate data
size = 1000
y1 = 2*np.sin(np.linspace(0, 100, size))
y2 = 3*np.cos(np.linspace(10, 100, size))
y3 = .1*np.linspace(20, 400, size)
target = y1+y2+y3
data = pd.DataFrame(np.c_[y1, y2, y3,target], columns=['v1', 'v2', 'v3', 'target'])

input_cols = ['target']
output_cols = ['target']
train = data.loc[:int(size*.8), :]
test = data.loc[int(size*.8):, :]
#%%

scaler = StandardScaler()
scaler.fit(train[input_cols])
train_normed = scaler.transform(train[input_cols])

train_normed.shape
type(train_normed)
test_normed = scaler.transform(test[input_cols])


#%%
train_normed.shape
test_normed.shape
def sample_batch(arr, n_batch, n_timestep):
    x , y1, y2 = [], [], []
    for i in range(n_batch):
        rand_idx = np.random.randint(0, len(arr)-n_timestep-1)
        x.append(range(rand_idx, rand_idx+n_timestep)) ##index
        y1.append( arr[rand_idx: rand_idx+n_timestep] ) ##input
        y2.append( arr[rand_idx+1: rand_idx+n_timestep+1] ) ##output
    return x, y1, y2

x, y1, y2 = sample_batch(train_normed, 2, 100)
x_test, y1_test, y2_test = sample_batch(test_normed, 2, 100)
fig, ax = plt.subplots(figsize=(22,10))
ax.plot(train_normed, 'k--', alpha=.4)
ax.plot(test_normed, 'r--', alpha=.4)
for i in range(len(y1)):
    ax.plot(x[i], y1[i])
    ax.plot(x_test[i], y1_test[i])
#%%
print(np.array(y1).shape)
print(np.array(y2).shape)
train.head()
test.head()
data.head()
#%%
#data_path =  "/dsaa/shared/users/wei_liu/work/llo/data/"
#df = pd.read_csv(data_path + 'df_small.csv')
#df.head()
#
#print(df['48h_score'].sum())
#df['48h_score'] =0
#df.fault_ts = pd.to_datetime(df.fault_ts)
#print(df.eq_nr.nunique())
#df['48h_score'] = df['48h'].apply(lambda x: x is not np.nan)
#df['48h_score'] = df['48h_score'].astype(int)
#df['48h_score'].sum()
#
#df.drop(['48h'], axis=1, inplace=True)
#df.dropna(axis=0, how='any', inplace=True)
#print(df.columns)
#
#for loco, mdf in df.groupby('eq_nr'):    
#    df.loc[mdf.index, 'time_diff'] = (mdf['fault_ts']-mdf['fault_ts'].shift()).fillna(0)
#
#df.time_diff = df.time_diff/pd.Timedelta('1 min')
#info_cols = ['eq_nr', 'fault_ts', 'f_6062_cnt']
#x_cols = ['oil_pres', 'oil_temp', 'rpm', 'water_press',
#          'manifold_temp', 'manifold_press', 'crankcase_press',
#          'turbo_speed', 'notch', 'time_diff', 'gross_hp']
#
#'''
#x_cols = ['oil_pres', 'oil_temp', 'rpm', 'water_press',
#           'crankcase_press', 'turbo_speed', 'notch','time_diff']
#'''
#
#y_col = ['48h_score']
#x_normed = [i+'_norm' for i in x_cols]



#%% show plots
fig, ax= plt.subplots(figsize=(22,10))
ax.plot(train[input_cols], marker='o')
ax.legend()
fig.show()
#%%
num_time_steps = 100
input_features = len(input_cols)
output_features = len(output_cols)
n_batch = 5
#%%
def scale_data(data):
    scaler = StandardScaler()
    n_timesteps = data.shape[1]
    n_features = data.shape[2]
    data_reshape = data.reshape(-1, n_features)
    scaler.fit(data_reshape)
    data_norm = scaler.transform(data_reshape)
    return np.array(data_norm).reshape(-1, n_timesteps, n_features), scaler
def inverse_scale_data(data, scaler):
    #n_timesteps = data.shape[1]
    n_features = data.shape[2]
    res = scaler.inverse_transform(data.reshape(-1, n_features))
    return res

def split_data_2(data, num_steps, input_cols, output_cols, norm=True):
    rows = data.shape[0]
    n = ((rows-1)//num_steps) * num_steps
    print('n = ', n)
    data = data.iloc[:n+1, :]
    assert n+1 <= rows
    x = data.iloc[:n][input_cols].as_matrix().reshape(-1, num_steps, len(input_cols))
    y = data.iloc[1:n+1][output_cols].as_matrix().reshape(-1, num_steps, len(output_cols))
    
    #data = data.reshape(-1, num_steps, n_features)
#    x = data[:n].reshape(-1, num_steps, input_features)
#    y = data[:n].reshape(-1, num_steps, output_features)
    if norm:
        x, scaler = scale_data(x)
        y = scaler.transform(y.reshape(-1, output_features)).reshape(-1, num_steps, output_features)
    return x, y


train_x, train_y = split_data_2(train, num_time_steps, input_cols, output_cols, False)
train_x_norm, scaler_x = scale_data(train_x)
train_y_norm, scaler_y = scale_data(train_y)

test_x, test_y = split_data_2(test, num_time_steps, input_cols, output_cols, False)
test_x_norm, scaler_x_test = scale_data(test_x)
test_y_norm, scaler_y_test = scale_data(test_y)

print(train_x_norm.shape)
print(train_y_norm.shape)

print(test_x_norm.shape)
print(test_y_norm.shape)

#%% split data
def get_batches(x, y, batch_size=39):
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size]

#%% lstm model
## build lstm rnn model
lstm_size = 256 # the number of hidden units in the LSTM cell
lstm_layers = 2 # how many layers of lstm, 2 is already good to use

learning_rate = 0.0005


tf.reset_default_graph()

inputs_ = tf.placeholder(tf.float32, [None, num_time_steps, input_features], name='inputs')
outputs_ = tf.placeholder(tf.float32, [None, num_time_steps, output_features], name='targets')

keep_prob = tf.placeholder(tf.float32, name='keep_prob')

#cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob) 

cells = []
for _ in range(lstm_layers):   
    cell = tf.nn.rnn_cell.LSTMCell(lstm_size, state_is_tuple=True)
    cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    cells.append(cell)

cells = tf.contrib.rnn.MultiRNNCell(cells,state_is_tuple=True)
#initial_state = cell.zero_state(batch_size, tf.float32)


outputs, state = tf.nn.dynamic_rnn(cells,
                                   inputs_,
                                   time_major=False, # default
                                   #sequence_length=[10]*batch_size,
                                   dtype=tf.float32)

predictions = tf.contrib.layers.fully_connected(outputs,
                                                num_outputs=output_features,
                                                activation_fn=None)

cost = tf.losses.mean_squared_error(outputs_, predictions)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

#%% train the model
epochs= 1000
validate_every_n = 300
#init = tf.global_variables_initializer()
saver = tf.train.Saver()
iters = []
mses = []
val_mses_mean = []
with tf.Session() as sess:   
    init = tf.global_variables_initializer()
    sess.run(init)

    for e in range(epochs):
        ## train
#        for b, (x, y) in enumerate(get_batches(train_x_norm, train_y_norm, 17), 1):            
        _, x, y = sample_batch(train_normed, n_batch, num_time_steps)
        start = time.time()
        feed = {inputs_: x,
                outputs_: y,
                keep_prob : .9
                }
        loss, _ = sess.run([cost,
                           optimizer], 
                           feed_dict=feed)

        if e % 10 == 0:    
            mse = cost.eval(feed_dict=feed)
            print('epoch = {}, mse = {}'.format(e, mse))
            mses.append(mse)
    # Save Model for Later
    saver.save(sess, "./lstm_test_2")
fig, ax = plt.subplots(figsize = (22, 10))
ax.plot(range(len(mses)), mses, 'b-', marker='o')
fig.show()
    
#%% test model
#split train data
def split_data(data, num_time_steps):
    rows = data.shape[0]
    n_batches = rows//num_time_steps
    return data[: n_batches*num_time_steps].reshape(-1, num_time_steps, 1)

res = split_data(train_normed, num_time_steps)
print(res.shape)
idxs = split_data(np.arange(train_normed.shape[0]).reshape(-1, 1), num_time_steps)

res_test = split_data(test_normed, num_time_steps)
print(res_test.shape)
idxs_test = split_data(np.arange(test_normed.shape[0]).reshape(-1, 1), num_time_steps)


fig, ax = plt.subplots(figsize=(22, 10))
for i in range(res.shape[0]):
    ax.plot(idxs[i], res[i])
for i in range(res_test.shape[0]):
    ax.plot(idxs_test[i], res_test[i])
fig.show()


#%%
test_preds = []
forward_steps = 20
with tf.Session() as sess:
    saver.restore(sess, "./lstm_test_2")
    y_preds = [[] for i in range(res.shape[0])]
    y_preds_test = [[] for i in range(res_test.shape[0])]
    for i in range(res.shape[0]):
        x_new = res[i].reshape(-1, num_time_steps, 1)
        for j in range(forward_steps):            
            #for i in range(test_x_norm.reshape(-1, num_time_steps, input_features).shape[0]):
            #x_new = test_x_norm.reshape(-1, num_time_steps, input_features)
            #x_new = test_x[i,:,:].reshape(-1, num_time_steps, input_features)
            y_pred = sess.run(predictions, feed_dict={inputs_:x_new,
                                                      keep_prob:1.0})
            
            y_preds[i].append(y_pred[0, -1, 0])
            tmp = np.r_[x_new[-1, 1:, :], np.array([y_pred[-1, -1, :]])]
            x_new = tmp.reshape(-1, num_time_steps, 1)



    for i in range(res_test.shape[0]):
        x_new_test = res_test[i].reshape(-1, num_time_steps, 1)
        for j in range(forward_steps):            

            y_pred_test = sess.run(predictions, feed_dict={inputs_:x_new_test,
                                                      keep_prob:1.0})
            
            y_preds_test[i].append(y_pred_test[0, -1, 0])
            tmp_test = np.r_[x_new_test[-1, 1:, :], np.array([y_pred_test[-1, -1, :]])]
            x_new_test = tmp_test.reshape(-1, num_time_steps, 1)
            
fig, ax = plt.subplots(figsize=(22, 10))
for i in range(res.shape[0]):
    ax.plot(idxs[i], res[i])
    ax.plot(range(idxs[i][-1, 0], idxs[i][-1, 0]+forward_steps), y_preds[i], 'b--')
for i in range(res_test.shape[0]):
    ax.plot(idxs_test[i], res_test[i])
    ax.plot(range(idxs_test[i][-1, 0], idxs_test[i][-1, 0]+forward_steps), y_preds_test[i], 'b--')
fig.show()


#%%
y_pred = np.array(y_pred).reshape(-1, output_features)
test_y_norm = test_y_norm.reshape(-1, output_features)
y_pred.shape
#%%
test_y_norm.shape
y_pred.shape

fig, (ax1) = plt.subplots(figsize=(22,10))
ax1.plot(test_y_norm[-10:], 'ro', ms=10, alpha=.3)
ax1.plot(y_pred[-10:], 'b.', ms=10, alpha=1)

#ax2.plot(y_pred)
#fig.show()

print(os.getcwd())
        