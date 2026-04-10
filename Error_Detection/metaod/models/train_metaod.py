# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 17:13:27 2020

@author: yuezh
"""

import os
import random
import pandas as pd
import numpy as np

from sklearn.utils import check_array
from sklearn.preprocessing import MinMaxScaler

from scipy.io import loadmat

from joblib import dump

from Error_Detection.metaod.models.utility import read_arff, fix_nan
from Error_Detection.metaod.models.gen_meta_features import generate_meta_features
from Error_Detection.metaod.models.core import MetaODClass

# read in performance table
# dir_path = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results/'
dir_path = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/results/'
roc_df = pd.read_csv(os.path.join(dir_path, 'performance_matrix_vus.csv'))

# trim the table
roc_mat = roc_df.to_numpy()
roc_mat_red = fix_nan(roc_mat[:, 1:].astype('float'))
print(roc_mat_red.shape)

# get statistics of the training data
n_datasets, n_configs = roc_mat_red.shape[0], roc_mat_red.shape[1]
data_headers = roc_mat[:, 0]
config_headers = roc_df.columns[1:]
print('data_headers:',data_headers)

save_dir = "/home/yyy/TSC/TSClean/AutoClean/Error_Detection/metaod/models/trained_models_0"
os.makedirs(save_dir, exist_ok=True) 
dump(config_headers, os.path.join(save_dir, "model_list.joblib"))

# %%

# build meta-features
meta_mat = np.zeros([n_datasets, 200])


data_direc = '/home/yyy/TSC/TSClean/AutoClean/Error_Detection/Train_Datasets_subsets/'

k = 0
for dataset in data_headers:
    dataset_name = dataset.split('_')[0]
    dataset_dir = os.path.join(data_direc, dataset_name)
    if not os.path.exists(dataset_dir):
        print(f"[Warning] Dataset directory not found: {dataset_dir}")
        continue

    # 查找所有 _val.csv 文件
    selected_files = [f for f in os.listdir(dataset_dir) if f.endswith('_val.csv')]
    if len(selected_files) == 0:
        print(f"[Warning] No test files found in {dataset_dir}")
        continue

    for file in selected_files:
        filepath = os.path.join(dataset_dir, file)

        df = pd.read_csv(filepath).dropna()
        X = df.iloc[:, 1:-1].values.astype(float)
        # print('X:',X.shape)
        meta_mat[k, :], meta_vec_names = generate_meta_features(X)
        print(k, file)
        k += 1


# 清洗 inf
meta_mat = np.where(np.isinf(meta_mat), np.nan, meta_mat)
# 1：全部填 0
# meta_mat = np.nan_to_num(meta_mat, nan=0.0)

# 2：列均值填充
col_mean = np.nanmean(meta_mat, axis=0)
inds = np.where(np.isnan(meta_mat))
meta_mat[inds] = np.take(col_mean, inds[1])

# use cleaned and transformed meta-features
meta_scalar = MinMaxScaler()
meta_mat_transformed = meta_scalar.fit_transform(meta_mat)
meta_mat_transformed = fix_nan(meta_mat_transformed)
dump(meta_scalar, os.path.join(save_dir, "meta_scalar.joblib"))
# %% train model

# split data into train and valid
seed = 0
full_list = list(range(n_datasets))
random.Random(seed).shuffle(full_list)
n_train = int(0.85 * n_datasets)

train_index = full_list[:n_train]
valid_index = full_list[n_train:]

train_set = roc_mat_red[train_index, :].astype('float64')
valid_set = roc_mat_red[valid_index, :].astype('float64')

train_meta = meta_mat_transformed[train_index, :].astype('float64')
valid_meta = meta_mat_transformed[valid_index, :].astype('float64')


train_set = np.nan_to_num(train_set, nan=0.0)
valid_set = np.nan_to_num(valid_set, nan=0.0)
train_meta = np.nan_to_num(train_meta, nan=0.0)
valid_meta = np.nan_to_num(valid_meta, nan=0.0)

print("NaN count:", np.isnan(train_meta).sum())
print(len(train_meta))

clf = MetaODClass(train_set, valid_performance=valid_set, n_factors=10, learning='sgd')

clf.train(n_iter=50, meta_features=train_meta, valid_meta=valid_meta,
        learning_rate=0.05, max_rate=0.9, min_rate=0.1, discount=1,
        n_steps=8)













# U = clf.user_vecs
# V = clf.item_vecs

# # # print(EMF.regr_multirf.predict(test_meta).shape)
# predicted_scores = clf.predict(valid_meta)
# predicted_scores_max = np.nanargmax(predicted_scores, axis=1)
# print()
# output transformer (for meta-feature) and the trained clf

clf_save = 'train_' + str(seed) + '.joblib'
dump(clf, os.path.join(save_dir, clf_save))

#%%
# # %%
# import pickle
# from metaod.models.core import MetaODClass

# if __name__ == "__main__":
#     # # code for standalone use
#     # t = Thing("foo")
#     # Thing.__module__ = "thing"
#     # t.save("foo.pickle")
#     # MetaODClass.__module__ = "metaod"
#     file = open('test.pk', 'wb')
#     pickle.dump(clf, file)

# # # file = open('rf.pk', 'wb')
# # # pickle.dump(clf.user_vecs, file)
