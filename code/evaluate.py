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