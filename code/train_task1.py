# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:06:51 2019
@author: yonghui yang
"""

import os,pdb
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
traindata = np.load('../data/traindata.npy').tolist()
valdata = np.load('../data/valdata.npy').tolist()
testdata = np.load('../data/testdata.npy').tolist()
user_items = np.load('../data/user_items.npy').tolist()
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
loss_txt = open(base_path+'loss.txt', 'w')
model_save_path = base_path + 'models/'
if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)
copyfile('train_task1.py', base_path+'model.py')


def get_1000_negatives(rating_all, negatives_sample=1000):
    '''
    random select 1000 negative samples for each user 
    '''
    user_negatives_1000 = {}
    for key in rating_all.keys():
        neg_list = itemset.symmetric_difference(rating_all[key])
        user_negatives_1000[key] = random.sample(neg_list, negatives_sample)
    return user_negatives_1000
user_negatives_1000 = get_1000_negatives(user_items)      

def get_train_adj_matrix(train_rating):
    '''
    get adjacent matrix of traindata#
    '''
    item_user_train = defaultdict(set)
    for key in train_rating.keys():
        for i in train_rating[key]:
            item_user_train[i].add(key)
    A_indexs = []
    A_values = []
    for x in train_rating.keys():
        len_u = len(train_rating[x])
        for i in range(len(train_rating[x])):
            y = train_rating[x][i] + user_count
            len_v = len(item_user_train[i])
            A_indexs.append([x,y])
            A_values.append(1/len_u)
            A_indexs.append([y,x])
            #A_values.append(1/len_u)
            A_values.append(1/len_v)
    return A_indexs, A_values

def get_bpr_data(rating_all, rating_train, item_count, neg_sample):
    '''
    get triple data [u,i,j] for training
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


def get_metrics(test_ratings, all_ratings, topk_list):
    '''
    compute valdata's ndcg for convergence metric
    '''    
    user_matrix, item_matrix = sess.run([final_user_emb,final_item_emb], 
                                feed_dict={item_attri_input:item_attributes_missing})
    hr_list = defaultdict(list)
    ndcg_list = defaultdict(list)
    hr_out = {}
    ndcg_out = {}
    for k in range(int(user_matrix.shape[0]/10000)+1):
        user_list = []
        pos_predictions = {}
        neg_predictions = {}
        start_user = k*10000
        end_user = min((k+1)*10000,user_matrix.shape[0])
        ratings = user_matrix[start_user:end_user,:].dot(item_matrix.T)
        for i in range(ratings.shape[0]):
            user_list.append(i)
            pos_predictions[i] = list(ratings[i,test_ratings[i+start_user]])
            neg_predictions[i] = list(ratings[i,user_negatives_1000[i+start_user]])   
        for topk in topk_list:
            hr,ndcg = evaluate.get_hr_ndcg(user_list, pos_predictions, neg_predictions, topk)
            hr_list[topk].append(hr*len(user_list))
            ndcg_list[topk].append(ndcg*len(user_list))
    for topk in topk_list:
        hr_out[topk] = round(sum(hr_list[topk])/user_matrix.shape[0], 4)
        ndcg_out[topk] = round(sum(ndcg_list[topk])/user_matrix.shape[0], 4)
        print('topk:', topk, 'hr:', hr_out[topk], 'ndcg:',ndcg_out[topk])
    return hr_out, ndcg_out



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


adjacent_matrix = tf.SparseTensor(indices=A_indexs, values=A_values, dense_shape=[user_count+item_count, user_count+item_count]) #[m+n, m+n]
initial_user_emb = tf.concat([user_emb,user_attri_emb],1) #[m, d1+d2]
initial_item_emb = tf.concat([item_emb,item_attri_emb],1) #[n, d1+d2]
feature_matrix_layer0 = tf.concat([initial_user_emb,initial_item_emb],0) #[m+n, d1+d2]
neighbor_matrix_layer1 = tf.sparse_tensor_dense_matmul(adjacent_matrix, feature_matrix_layer0)
feature_matrix_layer1 = tf.add(feature_matrix_layer0, neighbor_matrix_layer1)
neighbor_matrix_layer2 = tf.sparse_tensor_dense_matmul(adjacent_matrix, feature_matrix_layer1)
feature_matrix_layer2 = tf.add(feature_matrix_layer1, neighbor_matrix_layer2)
final_user_emb, final_item_emb = tf.split(feature_matrix_layer2,[user_count,item_count],0)


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


print('*****start train*****')
all_best_ckpt = []
all_max_hr = []
all_max_ndcg = []
for update_count in range(10):
    print('update=',update_count)
    loss_txt.write('update='+str(update_count)+'\n')
    maxhr, maxndcg, best_epoch = 0, 0, 0
    valhr,valndcg = [], []
    for epoch in range(epochs):
        # train part
        ### prepare [u,i,j] ###
        tt1 = time()
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
            sum_auc += _auc * u_list.shape[0]
            sum_loss1 += _loss1 * u_list.shape[0]
            sum_loss2 += _loss2 * u_list.shape[0]
            sum_loss += _loss * u_list.shape[0]
            sum_train += u_list.shape[0]           
        mean_auc = round(sum_auc/sum_train, 4)
        mean_loss1 = round(sum_loss1/sum_train, 4)
        mean_loss2 = round(sum_loss2/sum_train, 4)
        mean_loss = round(sum_loss/sum_train, 4)
        print('update:{:d},epoch:{:d}, train auc:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss:{:.4f}'
              .format(update_count,epoch,mean_auc,mean_loss1,mean_loss2,mean_loss))
        loss_txt.write('update:{:d},epoch:{:d}, train auc:{:.4f}, loss1:{:.4f}, loss2:{:.4f}, loss:{:.4f}'
              .format(update_count,epoch,mean_auc,mean_loss1,mean_loss2,mean_loss) + '\n')
        tt2 = time()
        print('train time:',round(tt2-tt1,4))
    
        #val part
        _hr,_ndcg = get_metrics(valdata, user_items, [10]) 
        hr = _hr[10]
        ndcg = _ndcg[10]
        valhr.append(hr)
        valndcg.append(ndcg)
        maxhr = max(maxhr, hr)
        maxndcg = max(maxndcg, ndcg)
        tt3 = time()
        print('val time:',round(tt3-tt2,4))    
        loss_txt.write('update:{:d},epoch:{:d}, hr:{:.4f}, ndcg:{:.4f}, train time:{:.4f}, val time:{:.4f}'
                       .format(update_count,epoch,hr,ndcg, tt2-tt1, tt3-tt2) + '\n\n')
        if valndcg[epoch] == maxndcg:
            best_epoch = epoch
            saver.save(sess, model_save_path+'update_'+str(update_count)+'epoch_'+str(epoch)+'.ckpt')        
        if update_count == 0 and epoch>40 and valndcg[epoch] <= valndcg[epoch-1] and valndcg[epoch-1] <= valndcg[epoch-2]:
            all_max_hr.append(maxhr)
            all_max_ndcg.append(maxndcg)
            all_best_ckpt.append(model_save_path+'update_'+str(update_count)+'epoch_'+str(best_epoch)+'.ckpt')
            saver.restore(sess, model_save_path+'update_'+str(update_count)+'epoch_'+str(best_epoch)+'.ckpt')
            _infer_attributes = sess.run(inferenced_item, 
                             feed_dict={item_attri_input: item_attributes_missing,
                                        item_missing_input: np.reshape(np.asarray(missing_item_list), [-1,1])})
            item_attributes_missing[missing_item_list] = _infer_attributes
            print('attributes has updated','\n')
            print('update:',update_count,'maxhr:',maxhr,'maxndcg',maxndcg)
            break

        elif update_count > 0 and epoch > 5 and valndcg[epoch] <= valndcg[epoch-1] and valndcg[epoch-1] <= valndcg[epoch-2]:
            all_max_hr.append(maxhr)
            all_max_ndcg.append(maxndcg)
            all_best_ckpt.append(model_save_path+'update_'+str(update_count)+'epoch_'+str(best_epoch)+'.ckpt')
            saver.restore(sess, model_save_path+'update_'+str(update_count)+'epoch_'+str(best_epoch)+'.ckpt')
            _infer_attributes = sess.run(inferenced_item, 
                             feed_dict={item_attri_input: item_attributes_missing,
                                        item_missing_input: np.reshape(np.asarray(missing_item_list), [-1,1])})
            item_attributes_missing[missing_item_list] = _infer_attributes
            print('attributes has updated','\n')
            print('update:',update_count,'maxhr:',maxhr,'maxndcg',maxndcg)
            break
        loss_txt.write('\n')
        
    if update_count > 1 and all_max_ndcg[update_count] <= all_max_ndcg[update_count-1] and all_max_ndcg[update_count-1] <= all_max_ndcg[update_count-2]:
        break
np.save(base_path+'all_best_ckpt.npy', all_best_ckpt)
print('********train over********')
print('maxhr=',max(all_max_hr),'maxndcg=',max(all_max_ndcg))
