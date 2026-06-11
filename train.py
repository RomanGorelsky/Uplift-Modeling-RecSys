import pandas as pd
from pathlib import Path
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from models import Causal_Model, Causal_Model_Mod
# from CJBPR import CJBPR
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
import datetime
import os
import itertools
# from dataset import dh_original, dh_personalized, ml_data

plotpath = "./results/"
if not os.path.isdir(plotpath):
    os.makedirs(plotpath)
def diff(list1, list2):
    return list(set(list2).difference(set(list1)))

def sparse_gather(indices, values, selected_indices, axis=0):
    """
    indices: [[idx_ax0, idx_ax1, idx_ax2, ..., idx_axk], ... []]
    values:  [ value1,                                 , ..., valuen]
    """
    mask = tf.equal(indices[:, axis][tf.newaxis, :], selected_indices[:, tf.newaxis])
    to_select = tf.where(mask)[:, 1]
    user_item = tf.gather(indices, to_select, axis=0)
    user = tf.gather(user_item, 0, axis=1)
    item = tf.gather(user_item, 1, axis=1)
    values = tf.gather(values, to_select, axis=0)
    return user, item, values


def count_freq(x):
    unique, counts = np.unique(x, return_counts=True)
    return np.asarray((unique, counts)).T

def predict_batches(model, user_array, item_array, batch_size=5000):
    dataset = tf.data.Dataset.from_tensor_slices((user_array, item_array))
    p_batches = []
    r_batches = []
    for u, i in dataset.batch(batch_size):
        _, p_batch, r_batch, _ = model((u, i), training=False)
        p_batches.append(p_batch.numpy())
        r_batches.append(r_batch.numpy())
    return np.concatenate(p_batches, axis=0), np.concatenate(r_batches, axis=0)

def preprocess_data(datapath, dataset):
    if dataset[-1] == 'd' or dataset[-1] == 'p':
        train_data = datapath / "data_train.csv"
        vali_data = datapath / "data_vali.csv"
        test_data = datapath / "data_test.csv"
        train_df = pd.read_csv(train_data)
        vali_df = pd.read_csv(vali_data)
        test_df = pd.read_csv(test_data)

        user_ids = np.sort(
            pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        item_ids = np.sort(
            pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
        item2item_encoded = {x: i for i, x in enumerate(item_ids)}
        train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
        train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
        vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
        vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
        test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
        test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
        num_users = len(user_ids)
        num_items = len(item_ids)
        if dataset == "d" or dataset == "p":
            num_times = len(train_df["idx_time"].unique().tolist())
        else: 
            num_times = 1
            train_df["idx_time"] = 0
            vali_df["idx_time"] = 0
            test_df["idx_time"] = 0
        train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated"]]
        return train_df, vali_df, test_df, num_users, num_items, num_times
    
    elif dataset == "f":
        dataset = pd.read_csv(datapath)
        dataset.rename(columns={"user_id": "idx_user", "item_id": "idx_item", "is_recommended": "treated", 
                                "timestamp": "idx_time", "label": "outcome", "user_total_taps": "frequency"}, inplace = True)
        dataset["personal_popular"] = dataset["frequency"]
        dataset["propensity"] = dataset["treated"].map({1: 0.9999, 0: 0.0001})
        dataset = dataset[["idx_user", "idx_item", "propensity", "treated", "outcome", "frequency", "personal_popular", "idx_time"]]
        train_df, test_df = train_test_split(dataset, test_size = 0.2, random_state = 42)
        # train_df, vali_df = train_test_split(train, test_size = 0.2, random_state = 42)

        user_ids = np.sort(
            pd.concat([train_df["idx_user"], test_df["idx_user"]]).unique().tolist())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        item_ids = np.sort(
            pd.concat([train_df["idx_item"], test_df["idx_item"]]).unique().tolist())
        item2item_encoded = {x: i for i, x in enumerate(item_ids)}
        train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
        train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
        # vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
        # vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
        test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
        test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
        num_users = len(user_ids)
        num_items = len(item_ids)
        num_times = 0
        train_df["idx_time"] = 0
        # vali_df["idx_time"] = 0
        test_df["idx_time"] = 0
        return train_df, None, test_df, num_users, num_items, num_times

    elif dataset == 't-ecd-small-m' or dataset == 't-ecd-small-r':
        train_data = datapath / "data_train.csv"
        vali_data = datapath / "data_vali.csv"
        test_data = datapath / "data_test.csv"

        train_df = pd.read_csv(train_data)
        vali_df = pd.read_csv(vali_data)
        test_df = pd.read_csv(test_data)

        user_ids = np.sort(
            pd.concat([train_df["idx_user"], vali_df["idx_user"], test_df["idx_user"]]).unique().tolist())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        item_ids = np.sort(
            pd.concat([train_df["idx_item"], vali_df["idx_item"], test_df["idx_item"]]).unique().tolist())
        item2item_encoded = {x: i for i, x in enumerate(item_ids)}
        train_df["idx_user"] = train_df["idx_user"].map(user2user_encoded)
        train_df["idx_item"] = train_df["idx_item"].map(item2item_encoded)
        vali_df["idx_user"] = vali_df["idx_user"].map(user2user_encoded)
        vali_df["idx_item"] = vali_df["idx_item"].map(item2item_encoded)
        test_df["idx_user"] = test_df["idx_user"].map(user2user_encoded)
        test_df["idx_item"] = test_df["idx_item"].map(item2item_encoded)
        num_users = len(user_ids)
        num_items = len(item_ids)
        num_times = len(train_df["idx_time"].unique().tolist())
        train_df = train_df[["idx_user", "idx_item", "outcome", "idx_time", "propensity", "treated", "personal_popular"]]
        return train_df, vali_df, test_df, num_users, num_items, num_times


def prepare_data(flag):
    dataset = flag.dataset
    data_path = None
    if flag.dataset == "1d":
        print("dunn_cate (original) 1 week is used.")
        data_path = Path("./CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/original_rp0.40")
        train_df, vali_df, test_df, _, _, _ = preprocess_data(data_path, dataset)
        train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
        train_df = train_df[train_df['idx_time'] == 0]
        df_merge = train_df[['idx_user', 'idx_item', 'personal_popular']]
        vali_df = vali_df[vali_df['idx_time'] == 0]
        test_df = test_df[test_df['idx_time'] == 0]
        vali_df = pd.merge(vali_df, df_merge, on=['idx_user', 'idx_item'], how='left')
        test_df = pd.merge(test_df, df_merge, on=['idx_user', 'idx_item'], how='left')
    elif flag.dataset == "10d":
        print("dunn_cate (original) 10 weeks is used.")
        data_path = Path("./CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/original_rp0.40")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "1p":
        print("dunn_cate (personalized) is 1 week used.")
        data_path = Path("./CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/rank_rp0.40_sf2.00_nr210")
        train_df, vali_df, test_df, _, _, _ = preprocess_data(data_path, dataset)
        train_df['personal_popular'] = train_df.groupby(['idx_user', 'idx_item'])['outcome'].transform('sum')
        train_df = train_df[train_df['idx_time'] == 0]
        df_merge = train_df[['idx_user', 'idx_item', 'personal_popular']]
        vali_df = vali_df[vali_df['idx_time'] == 0]
        test_df = test_df[test_df['idx_time'] == 0]
        vali_df = pd.merge(vali_df, df_merge, on=['idx_user', 'idx_item'], how='left')
        test_df = pd.merge(test_df, df_merge, on=['idx_user', 'idx_item'], how='left')
    elif flag.dataset == "10p":
        print("dunn_cate (personalized) 10 weeks is used.")
        data_path = Path("./CausalNBR/data/preprocessed/dunn_cat_mailer_10_10_1_1/10weeks/rank_rp0.40_sf2.00_nr210")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "ml":
        print("ML-100k is used")
        data_path = Path("./CausalNBR/data/synthetic/ML_100k_logrank100_offset5.0_scaling1.0")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "f":
        print("Finn-no is used")
        data_path = Path("./CausalNBR/data/finn-no/real_data.csv")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "t-ecd-small-m":
        print("T-ECD Small Marketplace is used")
        data_path = Path("./CausalNBR/data/t_ecd_small/marketplace")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    elif flag.dataset == "t-ecd-small-r":
        print("T-ECD Small Retail is used")
        data_path = Path("./CausalNBR/data/t_ecd_small/retail")
        train_df, vali_df, test_df, num_users, num_items, num_times = preprocess_data(data_path, dataset)
    
    train_df_positive = train_df[train_df["outcome"] > 0]
    counts = count_freq(train_df_positive['idx_item'].to_numpy())
    np_counts = np.zeros(num_items)
    np_counts[counts[:, 0].astype(int)] = counts[:, 1].astype(int)

    return train_df, vali_df, test_df, num_users, num_items, num_times, np_counts


def train_propensity(train_df, vali_df, flag, num_users, num_items, popular, num_run):

    if flag.prop_type == "orig":
        model = Causal_Model(num_users, num_items, flag, None, None, popular)
    elif flag.prop_type == "mod":
        model = Causal_Model_Mod(num_users, num_items, flag, None, None, popular)
    
    if flag.dataset == '1d' or flag.dataset == '1p':
        epochs = 10
    else:
        epochs = 10

    if flag.prop_train and num_run != 0:
        optim_val_car = 0
        # train_df = train_df[train_df["outcome"] > 0]
        # for epoch in range(flag.epoch):
        for epoch in range(epochs):
            print("Sampling negative items...", end=" ")
            train_items = train_df["idx_item"].to_numpy()
            j_list = np.random.randint(0, num_items, size=train_items.shape[0])
            same_mask = j_list == train_items
            while np.any(same_mask):
                j_list[same_mask] = np.random.randint(0, num_items, size=np.count_nonzero(same_mask))
                same_mask = j_list == train_items
            print("Done")
            j_list = j_list.astype(train_items.dtype, copy=False)
            train_data = tf.data.Dataset.from_tensor_slices((train_df["idx_user"].to_numpy(), train_items, j_list, train_df["outcome"].to_numpy()))
            with tqdm(total=len(train_df) // flag.batch_size + 1) as t:
                t.set_description('Training Epoch %i' % epoch)
                for user, item, item_j, value in train_data.shuffle(100).batch(flag.batch_size):
                    _ = model.propensity_train((user, item, item_j, value))
                    t.update()
            _, p_pred = predict_batches(
                model,
                vali_df["idx_user"].to_numpy(),
                vali_df["idx_item"].to_numpy(),
                batch_size=5000,
            )
            if 'propensity' in vali_df.columns:
                p_true = np.squeeze(vali_df["propensity"].to_numpy())
                p_pred = np.squeeze(p_pred)
                p_pred = (p_pred - np.min(p_pred)) / (np.max(p_pred) - np.min(p_pred))
                tau_res, _ = kendalltau(p_pred, p_true)
                val_obj = tau_res
                if abs(val_obj) > optim_val_car:
                    optim_val_car = val_obj
                    if not os.path.isdir(plotpath + '/' + flag.prop_add):
                        os.makedirs(plotpath + '/' + flag.prop_add)
                    model.save_weights(plotpath + flag.prop_add + "_" + str(num_run) + ".weights.h5")
                    print("Model saved!")
                    print(plotpath + flag.prop_add + "_" + str(num_run) + ".weights.h5")
            else:
                if not os.path.isdir(plotpath + '/' + flag.prop_add):
                    os.makedirs(plotpath + '/' + flag.prop_add)
                model.save_weights(plotpath + flag.prop_add + "_" + str(num_run) + ".weights.h5")
                print("Model saved!")
                print(plotpath + flag.prop_add + "_" + str(num_run) + ".weights.h5")

    else:
        sample_user = tf.constant([0])
        sample_item = tf.constant([0])
        _ = model((sample_user, sample_item))  # This builds all layers
        
    model.load_weights(plotpath + flag.prop_add + "_" + str(num_run) + ".weights.h5")
    return model

if __name__ == "__main__":
    pass