# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:06:51 2019
@author: Administrator
"""

import os,pdb
import pandas as pd
from collections import defaultdict
import tensorflow as tf
import numpy as np
import random
from time import time
import evaluate
from shutil import copyfile


### parameter ###
runid = 100
device_id = 0
dimension = 32
missing_rate = 0.9
learning_rate = 0.001
epochs = 100
batch_size = 1280 * 120
gama = 0.001
lamda1 = 0.01
lamda2 = 0.001
user_count = 138493
item_count = 27278


### read data ###
t1 = time()
user_items = np.load('../data/user_items.npy').tolist()
traindata = np.load('../data/traindata.npy').tolist()
valdata = np.load('../data/valdata.npy').tolist()
testdata = np.load('../data/testdata.npy').tolist()
A_indexs = np.load('../data/A_indexs.npy').tolist()
A_values = np.load('../data/A_values.npy').tolist()
item_attributes = np.load('../data/item_attributes.npy').astype(np.float32)
item_attributes_missing = np.load('../data/item_attributes_missing.npy')
existing_item_list = np.load('../data/existing_item_list.npy').tolist()
userset = set(range(user_count))
itemset = set(range(item_count))
missing_item_list = list(itemset.symmetric_difference(set(existing_item_list)))
t2 = time()
print('load data cost time:',round(t2-t1, 4))


base_path = '../task1/runid_'+str(runid)+'/'
if not os.path.exists(base_path):
    os.makedirs(base_path)
evaluate_txt = open(base_path+'evaluate.txt', 'w')
model_save_path = base_path + 'models/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
copyfile('test_task1.py', base_path+'testmodel.py')

                       
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
    att_loss = tf.nn.softmax_cross_entropy_with_logits(logits=item_infer, labels=item_gt_input)
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

saver = tf.train.Saver(val_dict, max_to_keep=0)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init)
sess.run(tf.local_variables_initializer())


### start test ###
all_best_ckpt = np.load(base_path+'all_best_ckpt.npy').tolist()
for k in range(len(all_best_ckpt)):
    print(all_best_ckpt[k])
    evaluate_txt.write('update_count:'+str(k)+'\n')
    if k == 0:
        saver.restore(sess, all_best_ckpt[0])
        item_attributes_predict = item_attributes_missing
        _user_matrix, _item_matrix = sess.run([final_user_emb, final_item_emb],
                                            feed_dict={item_attri_input:item_attributes_predict})
        hr_all, ndcg_all = evaluate.get_metrics_test(_user_matrix, _item_matrix, testdata, user_items, itemset, [10,20,30,40,50,60,70,80,90,100])
        for key in hr_all.keys():
            evaluate_txt.write('topk:{:d}, hr{:.4f}, ndcg:{:.4f}'.format(key, hr_all[key], ndcg_all[key]) + '\n')
    else:
        current_ckpt_file = all_best_ckpt[k-1]
        next_ckpt_file = all_best_ckpt[k]
        saver.restore(sess, current_ckpt_file)
        _infer_attributes = sess.run(inferenced_item, 
                             feed_dict={item_attri_input: item_attributes_missing,
                                        item_missing_input: np.reshape(np.asarray(missing_item_list), [-1,1])})
        item_attributes_missing[missing_item_list] = _infer_attributes
        item_attributes_predict = item_attributes_missing 
        saver.restore(sess, next_ckpt_file)
        _user_matrix, _item_matrix = sess.run([final_user_emb, final_item_emb],
                                            feed_dict={item_attri_input:item_attributes_predict})
        hr_all, ndcg_all = evaluate.get_metrics_test(_user_matrix, _item_matrix, testdata, user_items, itemset, [10,20,30,40,50,60,70,80,90,100])
        for key in hr_all.keys():
            evaluate_txt.write('topk:{:d}, hr{:.4f}, ndcg:{:.4f}'.format(key, hr_all[key], ndcg_all[key]) + '\n')
    evaluate_txt.write('****************')
