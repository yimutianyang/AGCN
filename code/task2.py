# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 12:12:02 2020

@author: Administrator
"""

import os,pdb
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random
from time import time
from sklearn.metrics import average_precision_score
from shutil import copyfile



### parameter ###
runid = 0
device_id = 3
dimension = 32
missing_rate = 0.9
learning_rate = 0.001
epochs = 300
batch_size = 1280 * 120
gama = 0.001
lamda1 = 0.01
lamda2 = 0.001
user_count = 138493
item_count = 27278


### read data ###
# generate from loaddata.py #
t1 = time()
user_items = np.load('../data/user_items.npy').tolist()
traindata = np.load('../data/traindata.npy').tolist()
valdata = np.load('../data/valdata.npy').tolist()
testdata = np.load('../data/testdata.npy').tolist()
A_indexs = np.load('../data/A_indexs.npy').tolist()
A_values = np.load('../data/A_values.npy').tolist()
user_negatives_1000 = np.load('../data/user_negatives_1000.npy').tolist()
item_attributes = np.load('../data/item_attributes.npy').astype(np.float32)
item_attributes_missing = np.load('../data/item_attributes_missing.npy')
existing_item_list = np.load('../data/existing_item_list.npy').tolist()
userset = set(range(user_count))
itemset = set(range(item_count))
missing_item_list = list(itemset.symmetric_difference(set(existing_item_list)))
t2 = time()
print('load data cost time:',round(t2-t1, 4))



base_path = './task2/runid_'+str(runid)+'/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
loss_txt = open(base_path+'loss.txt', 'w')
model_save_path = base_path + 'models/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
copyfile('task2.py', base_path+'task2.py')


def get_bpr_data(rating_all, rating_train, item_count, neg_sample):
    '''
    get triple data [u,i,j]
    '''
    t = []
    for u in rating_train.keys():
        for i in rating_train[u]:
            for _ in range(neg_sample):
                j = random.randint(0, item_count-1)
                while j in rating_all[u]:
                    j = random.randint(0, item_count-1)
                t.append([u,i,j])
    return np.reshape(np.asarray(t), [-1,3]) 


def get_rank_metric(pred, label):
    '''
    计算MAP
    '''
    ap_list = []
    for i in range(label.shape[0]):
        y_true = label[i]
        y_predict = pred[i]
        precision = average_precision_score(y_true, y_predict)
        ap_list.append(precision)
    mean_ap = sum(ap_list)/len(ap_list)
    return round(mean_ap,4)
    
# construct tensorflow graph
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
user_emb = tf.Variable(tf.random_normal([user_count, dimension], stddev=0.01), name='user_emb')
item_emb = tf.Variable(tf.random_normal([item_count, dimension], stddev=0.01), name='item_emb')
user_attri_emb = tf.Variable(tf.random_normal([user_count, 20], stddev=0.01), name='user_attri_emb')
item_trans_w = tf.Variable(tf.random_normal([20, 20], stddev=0.01), name='item_trans_w')
item_trans_b = tf.Variable(tf.zeros([20]), name='item_trans_b')
item_attri_input = tf.placeholder("float32", [None,20])
item_attri_emb = tf.matmul(item_attri_input, item_trans_w) #+ item_trans_b
inference_w = tf.Variable(tf.random_normal([52, 20], stddev=0.01), name='inference_w')
inference_b = tf.Variable(tf.zeros([20]), name='inference_b')


def gcn_model(propagation_layers):
    adjacent_matrix = tf.SparseTensor(indices=A_indexs, values=A_values, dense_shape=[user_count+item_count, user_count+item_count]) #[m+n, m+n]
    initial_user_emb = tf.concat([user_emb,user_attri_emb],1) #[m, d1+d2]
    initial_item_emb = tf.concat([item_emb,item_attri_emb],1) #[n, d1+d2]
    feature_matrix = tf.concat([initial_user_emb,initial_item_emb],0) #[m+n, d1+d2]
    k = 0     
    while k < propagation_layers:
        neighbor_matrix = tf.sparse_tensor_dense_matmul(adjacent_matrix, feature_matrix) #[m+n, d1+d2]
        feature_matrix = feature_matrix + neighbor_matrix
        k += 1
    return feature_matrix
feature_matrix = gcn_model(2)
final_user_emb, final_item_emb = tf.split(feature_matrix,[user_count,item_count],0)


# link prediction part
user_input = tf.placeholder("int32", [None, 1])
item_input = tf.placeholder("int32", [None, 1])
latent_user = tf.gather_nd(final_user_emb, user_input)
latent_item = tf.gather_nd(final_item_emb, item_input)
latent_mul = tf.multiply(latent_user, latent_item)
predictions = tf.sigmoid(tf.reduce_sum(latent_mul, 1, keepdims=True))

# attributes inference part
item_missing_input = tf.placeholder("int32", [None,1])
def get_inference():
    item_vector = tf.gather_nd(final_item_emb, item_missing_input)
    item_infer = tf.matmul(item_vector, inference_w) + inference_b
    out_infer = tf.sigmoid(item_infer)
    return out_infer 
inferenced_item = get_inference()

# link loss part
u_input = tf.placeholder("int32", [None, 1])
i_input = tf.placeholder("int32", [None, 1])
j_input = tf.placeholder("int32", [None, 1])
ua = tf.gather_nd(final_user_emb, u_input)
vi = tf.gather_nd(final_item_emb, i_input)
vj = tf.gather_nd(final_item_emb, j_input)
Rai = tf.reduce_sum(tf.multiply(ua, vi), 1, keepdims=True)
Raj = tf.reduce_sum(tf.multiply(ua, vj), 1, keepdims=True)
auc = tf.reduce_mean(tf.to_float((Rai-Raj)>0))
bprloss = -tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(Rai-Raj),1e-10,1.0)))
regulation = lamda1 * tf.reduce_mean(tf.square(ua)) + \
            lamda2 * tf.reduce_mean(tf.square(vi)+tf.square(vj))
loss1 = bprloss + regulation

# attributes loss part
item_existing_input = tf.placeholder("int32", [None,1])
item_gt_input = tf.placeholder("float32", [None,20])
def get_attributes_loss():
    item_vector = tf.gather_nd(final_item_emb, item_existing_input)
    item_infer = tf.sigmoid(tf.matmul(item_vector, inference_w) + inference_b)
    att_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=item_infer, labels=item_gt_input)
    out_loss = tf.reduce_mean(att_loss)
    return out_loss
loss2 = get_attributes_loss()
loss = loss1 + gama*loss2   
opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# start tensorflow session
init = tf.global_variables_initializer()
val_dict = {'user_emb':user_emb, 'item_emb':item_emb, 'user_attri_emb':user_attri_emb, 
            'item_trans_w':item_trans_w, 'item_trans_b':item_trans_b,
            'inference_w:':inference_w, 'inference_b':inference_b}

saver = tf.train.Saver(val_dict, max_to_keep=50)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)


print('*****start train*****')
MAP_list = []
max_map = 0
for epoch in range(epochs):
    # train part
    tt1 = time()
    sum_auc,sum_loss1,sum_loss2,sum_loss,sum_train = 0, 0, 0, 0, 0 
    t_train = get_bpr_data(user_items, traindata, item_count, 1)
    indexs = np.arange(t_train.shape[0])
    np.random.shuffle(indexs)
    sum_auc,sum_loss1,sum_loss2,sum_loss,sum_train = 0, 0, 0, 0, 0 
    for k in range(int(t_train.shape[0]/batch_size)+1):
        start_index = k*batch_size
        end_index = min(t_train.shape[0], (k+1)*batch_size)
        triple_data = t_train[indexs[start_index:end_index]]
        u_list, i_list, j_list = triple_data[:,0], triple_data[:,1], triple_data[:,2]
        _auc,_loss1,_loss2,_loss,_ = sess.run([auc,loss1,loss2,loss,opt], \
                                feed_dict={u_input:np.reshape(u_list,[-1,1]),
                                           i_input:np.reshape(i_list,[-1,1]), 
                                           j_input:np.reshape(j_list,[-1,1]),
                                           item_attri_input:item_attributes_missing,
                                           item_existing_input:np.reshape(np.asarray(existing_item_list),[-1,1]),
                                           item_gt_input:item_attributes[existing_item_list]})
        sum_auc += _auc * len(u_list)
        sum_loss1 += _loss1 * len(u_list)
        sum_loss2 += _loss2 * len(u_list)
        sum_loss += _loss * len(u_list)
        sum_train += len(u_list)           
    mean_auc = round(sum_auc/sum_train, 4)
    mean_loss1 = round(sum_loss1/sum_train, 4)
    mean_loss2 = round(sum_loss2/sum_train, 4)
    mean_loss = round(sum_loss/sum_train, 4)
    print('epoch:{:d}, train auc:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss:{:.4f}'
          .format(epoch,mean_auc,mean_loss1,mean_loss2,mean_loss))
    loss_txt.write('epoch:{:d}, train auc:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss:{:.4f}'
          .format(epoch,mean_auc,mean_loss1,mean_loss2,mean_loss) + '\n')
    tt2 = time()
    print('train time:',round(tt2-tt1,4))

    #val part
    _infer_attributes = sess.run(inferenced_item, 
                         feed_dict={item_attri_input: item_attributes_missing,
                                    item_missing_input: np.reshape(np.asarray(missing_item_list), [-1,1])})
    item_attributes_missing[missing_item_list] = _infer_attributes  # update attributes
    MAP = get_rank_metric(_infer_attributes, item_attributes[missing_item_list])
    print('epoch:',epoch,'map:',MAP)
    max_map = max(max_map, MAP)
    MAP_list.append(MAP)      
    loss_txt.write('epoch:'+str(epoch)+' testmap'+str(MAP))
    if epoch > 20 and MAP_list[epoch] < MAP_list[epoch-1] and MAP_list[epoch-1] < MAP_list[epoch-2]:
        print('*****early stop*****')        
print('********train over********')
print('maxmap=',max_map)