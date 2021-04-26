# -*- coding: utf-8 -*-
"""
Created on Fri Dec 20 20:06:51 2019
@author: yonghui yang
"""

import numpy as np
import math
from collections import defaultdict



def largest_indices(ary, n):
    """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def get_idcg(length):
    idcg = 0.0
    for i in range(length):
        idcg = idcg + math.log(2) / math.log(i + 2)
    return idcg


def get_hr_ndcg(_user_list,_pos_prediction,_neg_prediction,_topk):
    '''
    _user_list, dtype:list
    _pos_prediction, dict{user:[pos_list]}
    _neg_prediction, dict{user:[neg_list]}
    '''
    user_hr_list = []
    user_ndcg_list = []
    for i in range(len(_user_list)):
        tmp_user = _user_list[i]
        pos_length = len(_pos_prediction[tmp_user])
        target_length = min(_topk,pos_length) 
        
        if target_length > 0:
            pos_value = _pos_prediction[tmp_user]
            neg_value = _neg_prediction[tmp_user]
            pos_value.extend(neg_value)        
            tmp_prediction = np.asarray(pos_value)            
            del pos_value[-len(neg_value):]  ### important!!!            
            sort_index = np.argsort(tmp_prediction)[::-1] #sort index
            hit_value = 0
            dcg_value = 0        
            for idx in range(min(_topk,len(sort_index))):
                ranking = sort_index[idx]
                if ranking < pos_length:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx+2)   
            tmp_hr = round(hit_value/target_length,4)
            idcg = get_idcg(target_length)
            tmp_dcg = round(dcg_value/idcg,4)
            user_hr_list.append(tmp_hr)
            user_ndcg_list.append(tmp_dcg)
    mean_hr = sum(user_hr_list)/len(_user_list)
    mean_ndcg = sum(user_ndcg_list)/len(_user_list)
    return round(mean_hr,4),round(mean_ndcg,4)



def get_metrics_test(user_matrix, item_matrix, test_ratings, all_ratings, itemset, topk_list):
    '''
    compute hr&ndcg of all negative samples
    '''
    all_hr_list = defaultdict(list)
    all_ndcg_list = defaultdict(list)
    hr_out = {}
    ndcg_out = {}
    ratings = user_matrix.dot(item_matrix.T)
    for u in range(user_matrix.shape[0]):
        pos_index = list(test_ratings[u])
        pos_length = len(test_ratings[u])
        neg_index = list(itemset-all_ratings[u])
        pos_index.extend(neg_index)        
        pre_one=ratings[u][pos_index] 
        indices=largest_indices(pre_one, topk_list[-1])
        indices=list(indices[0])          
        for topk in topk_list:
            hit_value = 0
            dcg_value = 0  
            for idx in range(topk):
                ranking = indices[idx]
                if ranking < pos_length:
                    hit_value += 1
                    dcg_value += math.log(2) / math.log(idx+2) 
            target_length = min(topk,pos_length) 
            all_hr_list[topk].append(hit_value/target_length)
            idcg_value = get_idcg(target_length)
            all_ndcg_list[topk].append(dcg_value/idcg_value)
    for topk in topk_list:
        hr_out[topk] = round(sum(all_hr_list[topk])/user_matrix.shape[0], 4)
        ndcg_out[topk] = round(sum(all_ndcg_list[topk])/user_matrix.shape[0], 4)
        print('topk:', topk, 'hr:', hr_out[topk], 'ndcg:',ndcg_out[topk])
    return hr_out, ndcg_out


import pandas as pd
from collections import defaultdict
import numpy as np
import random
def read_data(path):
    ratings = pd.read_csv(path+'ratings.csv')
    movies = pd.read_csv(path+'movies.csv')
    movie_key = list(movies['movieId'].values)
    movie_id = list(range(27278))
    movie_dict = dict(zip(movie_key, movie_id))
    movie_features = list(movies['genres'].values)
    ratings = ratings.values[:,:2]
    user_items = defaultdict(set)
    for i in range(len(ratings)):
        user = int(ratings[i][0])-1
        item = movie_dict[int(ratings[i][1])]
        user_items[user].add(item)
    return user_items, movie_features
def encode_item(item_genres):
    all_genres = set()
    for i in range(len(item_genres)):
        for j in item_genres[i].split('|'):
            all_genres.add(j)
    genres_dict = dict(zip(list(all_genres), list(range(len(all_genres)))))
    item_attributes = np.zeros([item_count, len(all_genres)], dtype=np.int32)
    for i in range(len(item_genres)):
        for j in item_genres[i].split('|'):
            j_index = genres_dict[j]
            item_attributes[i][j_index] = 1
    return item_attributes
def split_data(rating_dict_input):
    traindata = {}
    valdata = {}
    testdata = {}
    for i in rating_dict_input.keys():
        traindata[i] = random.sample(rating_dict_input[i],max(1,int(len(rating_dict_input[i])*0.8)))
        val_test = rating_dict_input[i].symmetric_difference(traindata[i])
        valdata[i] = random.sample(val_test,int(len(val_test)/2))
        testdata[i] = list(val_test.symmetric_difference(valdata[i]))
    return traindata,valdata,testdata
def get_train_adj_matrix(train_rating,user_num,item_num):
    '''
    get train adajacent matrix
    '''
    item_user_train = defaultdict(set)
    for key in train_rating.keys():
        for i in traindata[key]:
            item_user_train[i].add(key)
    A_indexs = []
    A_values = []
    for key in train_rating.keys():
        print(key)
        x = key 
        len_u = len(train_rating[key])
        for i in range(len(train_rating[key])):
            y = train_rating[key][i] + user_num
            len_v = len(item_user_train[train_rating[key][i]])
            A_indexs.append([x,y])
            A_values.append(1/len_u)
            A_indexs.append([y,x])
            A_values.append(1/len_v)
    return A_indexs, A_values
def get_1000_negatives(rating_all, item_count, negatives_sample=1000):
    itemset = set(range(item_count))
    user_negatives_1000 = {}
    for key in rating_all.keys():
        neg_list = itemset.symmetric_difference(rating_all[key])
        user_negatives_1000[key] = random.sample(neg_list, negatives_sample)
    return user_negatives_1000
def get_missing_attributes(missing_rate, item_count):
    '''
    根据missing_rate随机删除item attributes
    1.初始化item attributes矩阵为0
    2.分别随机抽取存在不同attributes的产品列表
    3.根据2填充1
    '''
    itemset = set(range(item_count))
    item_attributes_missing = np.zeros([item_count, 20], dtype=np.float32)
    # padding existing profile
    existing_item_list = random.sample(list(itemset), int(item_count*(1-missing_rate)))
    item_attributes_missing[existing_item_list] = item_attributes[existing_item_list]
    # padding mean value for missing profiel
    missing_item_list = list(itemset.symmetric_difference(set(existing_item_list)))
    mean_attributes = np.mean(item_attributes[existing_item_list], axis=0) 
    item_attributes_missing[missing_item_list] = np.tile(mean_attributes, (len(missing_item_list),1)) 
    return item_attributes_missing, existing_item_list
if __name__ == 'main':
    data_path = '../ml-20m/'
    user_count = 138493
    item_count = 27278
    ### load ratings & attributes ###
    user_items, item_genres = read_data(data_path)
    np.save(data_path+'user_items.npy', user_items)
    np.save(data_path+'item_genres.npy', item_genres)    
    ### encode item attributes ###
    item_attributes = encode_item()
    np.save(data_path+'item_attributes.npy',item_attributes)
    ### split train/val/test ###    
    traindata,valdata,testdata = split_data(user_items)
    np.save(data_path+'traindata.npy',traindata)
    np.save(data_path+'valdata.npy',valdata)
    np.save(data_path+'testdata.npy',testdata)    
    ### get adjacent matrix ###
    A_indexs, A_values = get_train_adj_matrix(traindata, user_count, item_count)
    np.save(data_path+'A_indexs.npy',A_indexs)
    np.save(data_path+'A_values.npy',A_values)    
    ### get 1000 negative samples ###
    user_negatives_1000 = get_1000_negatives(user_items)
    np.save('user_negatives_1000.npy',user_negatives_1000)
