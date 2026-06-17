import argparse
from train import prepare_data, train_propensity, predict_batches
from train import plotpath, Causal_Model
from baselines import DLMF, DLMF_Mod, DLMF_DR, PopularBase, MF, CausalNeighborBase
import numpy as np
# from CJBPR import CJBPR
import tensorflow as tf
from evaluator import Evaluator
from scipy.stats import kendalltau
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import gaussian_kde
import pickle
import os
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--dimension", default=128, type=int, help="number of features per user/item.")
parser.add_argument("--estimator_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Propensity and Relevance estimators")
parser.add_argument("--embedding_layer_units",
                    default=[256, 128, 64],
                    type=list,
                    help="number of nodes each layer for shared embedding layer.")
parser.add_argument("--click_layer_units",
                    default=[64, 32, 16, 8],
                    type=list,
                    help="number of nodes each layer for MLP layers in Click estimators")
parser.add_argument("--epoch", default=25, type=int,
                    help="Number of epochs in the training")
parser.add_argument("--prop_type", default="mod", type=str,
                    help="Start the training of PropCare")
parser.add_argument("--rec_type", default="orig", type=str,
                    help="Version of Recommender used")
parser.add_argument("--prop_train", default=False, type=bool,
                    help="Start the training of PropCare")
parser.add_argument("--rec_train", default=False, type=bool,
                    help="Start the training of Recommender")
parser.add_argument("--continue_rec_train", default=False, type=bool,
                    help="Continue the training of Recommender")
parser.add_argument("--lambda_1", default=10.0, type=float,
                    help="weight for popularity loss.")
parser.add_argument("--lambda_2", default=0.1, type=float,
                    help="weight for relavance loss.")
parser.add_argument("--lambda_3", default=0.1, type=float,
                    help="weight for propensity2 loss.")
parser.add_argument("--dataset", default='d', type=str,
                    help="the dataset used")
parser.add_argument("--batch_size", default=5096, type=int,
                    help="the batch size")
parser.add_argument("--repeat", default=1, type=int,
                    help="how many time to run the model")
parser.add_argument("--add", default='default', type=str,
                    help="additional information")
parser.add_argument("--prop_add", default='', type=str,
                    help="additional information for PropCare")
parser.add_argument("--rec_add", default='', type=str,
                    help="additional information for Recommender")
parser.add_argument("--p_weight", default=0.4, type=float,
                    help="weight for p_loss")
parser.add_argument("--r_weight", default=0.4, type=float,
                    help="weight for r_loss")
flag = parser.parse_args()


def main(flag=flag):
    # for phi_t in [0.1, 0.2, 0.3]:
    #     for th in [0.5, 0.6, 0.7]:
    # for b in [1, 2, 5, 10, 20, 50]:

    if flag.dataset == "t-ecd-small-short-r":
        flag.prop_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/short/' 
        if flag.rec_type != 'rel':
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/' + flag.rec_type + '/short/'
        else:
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/orig/short/'
    elif flag.dataset == "t-ecd-small-long-r":
        flag.prop_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/long/'
        if flag.rec_type != 'rel':
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/' + flag.rec_type + '/long/'
        else:
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/orig/long/'
    else:
        flag.prop_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/'
        if flag.rec_type != 'rel':
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/' + flag.rec_type + '/'
        else:
            flag.rec_add = flag.add + '/' + flag.prop_type + '/' + flag.dataset[-1] + '/orig/'

    cp10list_pred = []
    cp100list_pred = []
    cdcglist_pred = []

    cp10list_pred_freq = []
    cp100list_pred_freq = []
    cdcglist_pred_freq = []

    cp10list_pred_freqi = []
    cp100list_pred_freqi = []
    cdcglist_pred_freqi = []

    cp10list_pred_frequ = []
    cp100list_pred_frequ = []
    cdcglist_pred_frequ = []

    cp10list_rel = []
    cp100list_rel = []
    cdcglist_rel = []

    cp10list_pop = []
    cp100list_pop = []
    cdcglist_pop = []

    cp10list_pers_pop = []
    cp100list_pers_pop = []
    cdcglist_pers_pop = []

    ndcglist_rel = []
    ndcglist_pred = []
    ndcglist_pred_freq = []
    ndcglist_pred_freqi = []
    ndcglist_pred_frequ = []
    ndcglist_pop = []
    ndcglist_pers_pop = []

    recalllist_rel = []
    recalllist_pred = []
    recalllist_pred_freq = []
    recalllist_pred_freqi = []
    recalllist_pred_frequ = []
    recalllist_pop = []
    recalllist_pers_pop = []

    precisionlist_rel = []
    precisionlist_pred = []
    precisionlist_pred_freq = []
    precisionlist_pred_freqi = []
    precisionlist_pred_frequ = []
    precisionlist_pop = []
    precisionlist_pers_pop = []

    dirk_list_pred = []
    dirk_list_pred_freq = []
    dirk_list_pred_freqi = []
    dirk_list_pred_frequ = []
    dirk_list_rel = []
    dirk_list_pop = []
    dirk_list_pers_pop = []

    dirk_coverage_list_pred = []
    dirk_coverage_list_pred_freq = []
    dirk_coverage_list_pred_freqi = []
    dirk_coverage_list_pred_frequ = []
    dirk_coverage_list_rel = []
    dirk_coverage_list_pop = []
    dirk_coverage_list_pers_pop = []

    rcau_list_pred = []
    rcau_list_pred_freq = []
    rcau_list_pred_freqi = []
    rcau_list_pred_frequ = []
    rcau_list_rel = []
    rcau_list_pop = []
    rcau_list_pers_pop = []

    random_seed = int(240)

    for num_run in range(flag.repeat):
        train_df, vali_df, test_df, num_users, num_items, num_times, popular = prepare_data(flag)
        random_seed += 1
        tf.random.set_seed(
            random_seed
        )

        if flag.dataset != "f":
            model = train_propensity(train_df, vali_df, flag, num_users, num_items, popular, num_run)

            if flag.dataset[-1] == 'd' or flag.dataset[-1] == 'p':
                opt_scale = 0.25
                opt_add = 0.5
                # opt_epsilon = 0.7
                # opt_c = 0.8
            if flag.dataset == 't-ecd-small-short-m' or flag.dataset == 't-ecd-small-long-m':
                opt_scale = 0.1
                opt_add = 0.0
                # opt_epsilon = 0.5
                # opt_c = 0.9
            if flag.dataset == 't-ecd-small-short-r' or flag.dataset == 't-ecd-small-long-r':
                opt_scale = 0.1
                opt_add = 0.0
                # opt_epsilon = 0.5
                # opt_c = 0.9
            else:
                opt_scale = 0.5
                opt_add = 0.25

            # start_scales, end_scales = 0.1, 0.9
            # start_adds, end_adds = 0, 1
            # start_epsilons, end_epsilons = 0.1, 0.9
            # start_cs, end_cs = 0.1, 0.9

            p_pred, r_pred = predict_batches(
                    model,
                    train_df["idx_user"].to_numpy(),
                    train_df["idx_item"].to_numpy(),
                    batch_size=5000,
                )

            p_pred_t = opt_scale * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
            p_pred_t = np.clip((p_pred_t + opt_add), 0.0, 1.0)

            r_pred_t = opt_scale * ((r_pred - np.mean(r_pred))/ (np.std(r_pred)))
            r_pred_t = np.clip((r_pred_t + opt_add), 0.0, 1.0)

            # p_pred_t = p_pred * opt_c
            # p_pred_t = np.clip(p_pred_t, 0.0001, 0.9999)

            # t_pred_t = np.where(p_pred_t >= opt_epsilon, 1.0, 0.0)
            # max_f = f1_score(train_df['treated'], t_pred_t)
            # roc_max = roc_auc_score(train_df['treated'], t_pred_t)
            # print('Initial F1 score', max_f)
            # print('Initial ROC-AUC', roc_max)

            # for _ in range(5):
            #     scales = np.linspace(start_scales, end_scales, 3)
            #     adds = np.linspace(start_adds, end_adds, 3)
            #     epsilons = np.linspace(start_epsilons, end_epsilons, 3)
            #     cs = np.linspace(start_cs, end_cs, 3)

            #     for scale in scales:
            #         for add in adds:
            #             for epsilon in epsilons:
            #                 for c in cs:
            #                     p_pred_t = scale * ((p_pred - np.mean(p_pred))/ (np.std(p_pred)))
            #                     p_pred_t = np.clip((p_pred_t + add), 0.0, 1.0)
            #                     p_pred_t = p_pred * c
            #                     p_pred_t = np.clip(p_pred_t, 0.0001, 0.9999)
            #                     t_pred_t = np.where(p_pred_t >= epsilon, 1.0, 0.0)
            #                     f_score = f1_score(train_df['treated'], t_pred_t)
            #                     roc = roc_auc_score(train_df['treated'], t_pred_t)
            #                     if f_score > max_f and roc > roc_max:
            #                         max_f = f_score
            #                         roc_max = roc
            #                         opt_c = c
            #                         opt_scale = scale
            #                         opt_add = add
            #                         opt_epsilon = epsilon

            #     if opt_c <= cs[1]:
            #         end_cs /= 2
            #     else:
            #         start_cs *= 2
                
            #     if opt_scale <= scales[1]:
            #         end_scales /= 2
            #     else:
            #         start_scales *= 2

            #     if opt_add <= adds[1]:
            #         end_adds /= 2
            #     else:
            #         start_adds *= 2
                
            #     if opt_epsilon <= epsilons[1]:
            #         end_epsilons /= 2
            #     else:
            #         start_epsilons *= 2

            #     print('-' * 20)
            #     print('Max F1 score: ', max_f)
            #     print('Max ROC-AUC: ', roc_max)
            #     print('Optimal scale: ', opt_scale)
            #     print('Optimal add: ', opt_add)
            #     print('Optimal epsilon: ', opt_epsilon)
            #     print('Optimal c: ', opt_c)

            # break

            if flag.dataset[-1] == "d":
                flag.thres = 0.7
                opt_c = 0.9
                lr = 0.001
                cap = 0.03
                rf = 0.01
                itr = 20e6
                phi = 0.1
                flag.rel_thresh = 0.5

            if flag.dataset[-1] == "p":
                flag.thres = 0.7
                opt_c = 0.8
                lr = 0.001
                cap = 0.5
                rf = 0.001
                itr = 70e6
                phi = 0.1
                flag.rel_thresh = 0.6

            if flag.dataset == 't-ecd-small-short-m' or flag.dataset == 't-ecd-small-long-m':
                flag.thres = 0.5 # 0.1
                opt_c = 0.9
                lr = 0.001
                cap = 0.3
                rf = 0.005
                itr = 150e6
                phi = 0.1
                flag.rel_thresh = 0.6
                beta = 10
            
            if flag.dataset == 't-ecd-small-short-r' or flag.dataset == 't-ecd-small-long-r':
                flag.thres = 0.5 # 0.1
                opt_c = 0.9
                lr = 0.001
                cap = 0.3
                rf = 0.005
                itr = 500e6
                phi = 0.3
                flag.rel_thresh = 0.7
                beta = 10

            if flag.dataset == "ml":
                opt_c = 0.2
                flag.thres = 0.65
                lr = 0.001
                cap = 0.3
                rf = 0.1
                itr = 5e5
                phi = 0.1
                flag.rel_thresh = 0.7
            
            p_pred = p_pred * opt_c
            r_pred = r_pred * opt_c

            t_pred = np.where(p_pred_t >= flag.thres, 1.0, 0.0)
            rel_pred = np.where(r_pred_t >= flag.rel_thresh, 1.0, 0.0)
            
            train_df["propensity"] = np.clip(p_pred, 0.0001, 0.9999)
            train_df["relevance"] = np.clip(r_pred, 0.0001, 0.9999)
            train_df["treated"] = t_pred
            train_df["relevant"] = rel_pred
        
        elif flag.dataset == "f":
            lr = 0.001
            cap = 0.5
            rf = 0.001
            itr = 5e5

        if flag.rec_type == "orig":
            recommender = DLMF(num_users, num_items, capping_T = cap, 
                            capping_C = cap, learn_rate = lr, reg_factor = rf)
            
        if flag.rec_type == "rel":
            recommender = DLMF(num_users, num_items, capping_T = cap, 
                            capping_C = cap, learn_rate = lr, reg_factor = rf, use_relevance=True, coeff_beta=beta)
        
        elif flag.rec_type == "mod":
            recommender = DLMF_Mod(num_users, num_items, capping_T = cap, 
                            capping_C = cap, learn_rate = lr, reg_factor = rf)
            
        elif flag.rec_type == "dr":
            recommender = DLMF_DR(num_users, num_items, capping_T = cap, 
                            capping_C = cap, learn_rate = lr, reg_factor = rf)
        
        elif flag.rec_type == "gbc":
            recommender = GradientBoostingClassifier(learning_rate = lr, random_state = 42)
        
        if flag.rec_train:
            if flag.rec_type == "orig" or flag.rec_type == "rel":
                if flag.continue_rec_train:
                    with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_weights.pkl", "rb") as f:
                        saved_state = pickle.load(f)
                    recommender.__dict__.update(saved_state)
                    print("DLMF continued training started!")
                recommender.train(train_df, plotpath + flag.rec_add, num_run, iter=itr)
            elif flag.rec_type == "mod":
                if flag.continue_rec_train:
                    with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_mod_weights.pkl", "rb") as f:
                        saved_state = pickle.load(f)
                    recommender.__dict__.update(saved_state)
                    print("DLMF_Mod continued training started!")
                recommender.train(train_df, plotpath + flag.rec_add, num_run, phi, iter=itr)
            elif flag.rec_type == "dr":
                if flag.continue_rec_train:
                    with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_dr_weights.pkl", "rb") as f:
                        saved_state = pickle.load(f)
                    recommender.__dict__.update(saved_state)
                    print("DLMF_DR continued training started!")
                recommender.train(train_df, plotpath + flag.rec_add, iter=itr)
            elif flag.rec_type == "gbc":
                popularity = train_df[train_df.outcome>0]["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                train_df = train_df.merge(popularity, on="idx_item", how="left")
                train_df['popularity'] = (train_df['popularity'] - np.min(train_df['popularity'])) \
                                            / (np.max(train_df['popularity']) - np.min(train_df['popularity']))
                train_df['popularity'] = train_df['popularity'].fillna(0)

                recommender.fit(train_df.drop(['outcome'], axis=1),
                                train_df['outcome'])
        
        else:
            if flag.rec_type == "orig" or flag.rec_type == "rel":
                with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_weights.pkl", "rb") as f:
                    saved_state = pickle.load(f)
                recommender.__dict__.update(saved_state)
                print("DLMF weights loaded successfully!")
            elif flag.rec_type == "mod":
                with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_mod_weights.pkl", "rb") as f:
                    saved_state = pickle.load(f)
                recommender.__dict__.update(saved_state)
                print("DLMF_Mod weights loaded successfully!")
            elif flag.rec_type == "dr":
                with open(plotpath + flag.rec_add + str(num_run) + "_" + "dlmf_dr_weights.pkl", "rb") as f:
                    saved_state = pickle.load(f)
                recommender.__dict__.update(saved_state)
                print("DLMF_DR weights loaded successfully!")

        cp10_tmp_list_pred = []
        cp100_tmp_list_pred = []
        cdcg_tmp_list_pred = []

        cp10_tmp_list_pred_freq = []
        cp100_tmp_list_pred_freq = []
        cdcg_tmp_list_pred_freq = []

        cp10_tmp_list_pred_freqi = []
        cp100_tmp_list_pred_freqi = []
        cdcg_tmp_list_pred_freqi = []

        cp10_tmp_list_pred_frequ = []
        cp100_tmp_list_pred_frequ = []
        cdcg_tmp_list_pred_frequ = []

        cp10_tmp_list_rel = []
        cp100_tmp_list_rel = []
        cdcg_tmp_list_rel = []

        cp10_tmp_list_pop = []
        cp100_tmp_list_pop = []
        cdcg_tmp_list_pop = []

        cp10_tmp_list_pers_pop = []
        cp100_tmp_list_pers_pop = []
        cdcg_tmp_list_pers_pop = []
        
        ndcg_tmp_list_rel = []
        ndcg_tmp_list_pred = []
        ndcg_tmp_list_pred_freq = []
        ndcg_tmp_list_pred_freqi = []
        ndcg_tmp_list_pred_frequ = []
        ndcg_tmp_list_pop = []
        ndcg_tmp_list_pers_pop = []

        recall_tmp_list_rel = []
        recall_tmp_list_pred = []
        recall_tmp_list_pred_freq = []
        recall_tmp_list_pred_freqi = []
        recall_tmp_list_pred_frequ = []
        recall_tmp_list_pop = []
        recall_tmp_list_pers_pop = []

        precision_tmp_list_rel = []
        precision_tmp_list_pred = []
        precision_tmp_list_pred_freq = []
        precision_tmp_list_pred_freqi = []
        precision_tmp_list_pred_frequ = []
        precision_tmp_list_pop = []
        precision_tmp_list_pers_pop = []

        dirk_tmp_list_pred = []
        dirk_tmp_list_pred_freq = []
        dirk_tmp_list_pred_freqi = []
        dirk_tmp_list_pred_frequ = []
        dirk_tmp_list_rel = []
        dirk_tmp_list_pop = []
        dirk_tmp_list_pers_pop = []

        dirk_coverage_tmp_list_pred = []
        dirk_coverage_tmp_list_pred_freq = []
        dirk_coverage_tmp_list_pred_freqi = []
        dirk_coverage_tmp_list_pred_frequ = []
        dirk_coverage_tmp_list_rel = []
        dirk_coverage_tmp_list_pop = []
        dirk_coverage_tmp_list_pers_pop = []

        rcau_tmp_list_pred = []
        rcau_tmp_list_pred_freq = []
        rcau_tmp_list_pred_freqi = []
        rcau_tmp_list_pred_frequ = []
        rcau_tmp_list_rel = []
        rcau_tmp_list_pop = []
        rcau_tmp_list_pers_pop = []

        if flag.dataset[-1] == 'd' or flag.dataset[-1] == 'p' \
            or "t-ecd-small" in flag.dataset:
            for t in range(num_times):
                test_df_t = test_df[test_df["idx_time"] == t]
                p_pred_test, r_pred_test = predict_batches(
                    model,
                    test_df_t["idx_user"].to_numpy(),
                    test_df_t["idx_item"].to_numpy(),
                    batch_size=5000,
                )
                p_pred_test_t = opt_scale * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + opt_add), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)

                p_pred_test = p_pred_test * opt_c
                r_pred_test = r_pred_test * opt_c 

                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)

                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = opt_scale * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + opt_add), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome_estimate"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                train_df = train_df[train_df.outcome>0]
                popularity = train_df["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
                test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                                            / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
                test_df_t['popularity'] = test_df_t['popularity'].fillna(0)
                test_df_t['frequency'] = test_df_t['personal_popular']
                test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

                print("Prediction started")

                test_df_t["pred"] = recommender.predict(test_df_t)
                test_df_t["pred_freq"] = recommender.predict_freq(test_df_t)
                test_df_t["pred_freqi"] = recommender.predict_freqi(test_df_t)
                test_df_t["pred_frequ"] = recommender.predict_frequ(test_df_t)

                print("Evaluation started")

                evaluator = Evaluator()

                test_df_s = evaluator.get_sorted(test_df_t)
                test_df_sf = evaluator.get_sorted(test_df_t, "pred_freq")
                test_df_sfi = evaluator.get_sorted(test_df_t, "pred_freqi")
                test_df_sfu = evaluator.get_sorted(test_df_t, "pred_frequ")
                test_df_r = evaluator.get_sorted(test_df_t, "relevance_estimate")
                test_df_p = evaluator.get_sorted(test_df_t, "popularity")
                test_df_pp = evaluator.get_sorted(test_df_t, "personal_popular")

                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'CPrecS', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'CPrecS', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'CDCGS', 100000))

                cp10_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'CPrecSF', 10))
                cp100_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'CPrecSF', 100))
                cdcg_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'CDCGSF', 100000))

                cp10_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'CPrecSFI', 10))
                cp100_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'CPrecSFI', 100))
                cdcg_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'CDCGSFI', 100000))
                
                cp10_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'CPrecSFU', 10))
                cp100_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'CPrecSFU', 100))
                cdcg_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'CDCGSFU', 100000))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'CDCGR', 100000))

                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'CDCGP', 100000))

                cp10_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'CPrecPP', 10))
                cp100_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'CPrecPP', 100))
                cdcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'CDCGPP', 100000))

                print("Causal Metrics done")

                # ==================== DIR@K (Debiased Incremental Response at K) ====================
                ms, cs = evaluator.evaluate(test_df_s, 'DIR@KS', 10)
                dirk_tmp_list_pred.append(ms)
                dirk_coverage_tmp_list_pred.append(cs)

                msf, csf = evaluator.evaluate(test_df_sf, 'DIR@KSF', 10)
                dirk_tmp_list_pred_freq.append(msf)
                dirk_coverage_tmp_list_pred_freq.append(csf)
                
                msfi, csfi = evaluator.evaluate(test_df_sfi, 'DIR@KSFI', 10)
                dirk_tmp_list_pred_freqi.append(msfi)
                dirk_coverage_tmp_list_pred_freqi.append(csfi)
                
                msfu, csfu = evaluator.evaluate(test_df_sfu, 'DIR@KSFU', 10)
                dirk_tmp_list_pred_frequ.append(msfu)
                dirk_coverage_tmp_list_pred_frequ.append(csfu)

                msr, csr = evaluator.evaluate(test_df_r, 'DIR@KR', 10)
                dirk_tmp_list_rel.append(msr)
                dirk_coverage_tmp_list_rel.append(csr)

                msp, csp = evaluator.evaluate(test_df_p, 'DIR@KP', 10)
                dirk_tmp_list_pop.append(msp)
                dirk_coverage_tmp_list_pop.append(csp)

                mspp, cspp = evaluator.evaluate(test_df_pp, 'DIR@KPP', 10)
                dirk_tmp_list_pers_pop.append(mspp)
                dirk_coverage_tmp_list_pers_pop.append(cspp)

                rcau_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'RCAU@KS', 10))
                rcau_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'RCAU@KSF', 10))
                rcau_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'RCAU@KSFI', 10))
                rcau_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'RCAU@KSFU', 10))
                rcau_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'RCAU@KR', 10))
                rcau_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'RCAU@KP', 10))
                rcau_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'RCAU@KPP', 10))

                print("New causal metrics done")

                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'NDCGS', 10))
                ndcg_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'NDCGSF', 10))
                ndcg_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'NDCGSFI', 10))
                ndcg_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'NDCGSFU', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'NDCGP', 10))
                ndcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'NDCGPP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'RecallS', 10))
                recall_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'RecallSF', 10))
                recall_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'RecallSFI', 10))
                recall_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'RecallSFU', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'RecallP', 10))
                recall_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'RecallPP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_r, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_s, 'PrecisionS', 10))
                precision_tmp_list_pred_freq.append(evaluator.evaluate(test_df_sf, 'PrecisionSF', 10))
                precision_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_sfi, 'PrecisionSFI', 10))
                precision_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_sfu, 'PrecisionSFU', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_p, 'PrecisionP', 10))
                precision_tmp_list_pers_pop.append(evaluator.evaluate(test_df_pp, 'PrecisionPP', 10))

                print('Regular Metrics done')

                if flag.dataset[-1] == 'd' or flag.dataset[-1] == 'p':

                    kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                    spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                    pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                    kendall_score_freq = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred_freq', 'relevance_estimate')
                    spearman_score_freq = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred_freq', 'relevance_estimate')
                    pos_diff_freq = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred_freq', 'relevance_estimate')

                    kendall_score_freqi = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred_freqi', 'relevance_estimate')
                    spearman_score_freqi = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred_freqi', 'relevance_estimate')
                    pos_diff_freqi = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred_freqi', 'relevance_estimate')

                    kendall_score_frequ = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred_frequ', 'relevance_estimate')
                    spearman_score_frequ = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred_frequ', 'relevance_estimate')
                    pos_diff_frequ = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred_frequ', 'relevance_estimate')

                    print(f"Kendall Tau: {kendall_score:.4f}")
                    print(f"Spearman Rho: {spearman_score:.4f}")
                    print(f"Average Rank Position Difference: {pos_diff:.4f}")

                    print(f"Kendall Tau (freq): {kendall_score_freq:.4f}")
                    print(f"Spearman Rho (freq): {spearman_score_freq:.4f}")
                    print(f"Average Rank Position Difference (freq): {pos_diff_freq:.4f}")

                    print(f"Kendall Tau (freqi): {kendall_score_freqi:.4f}")
                    print(f"Spearman Rho (freqi): {spearman_score_freqi:.4f}")
                    print(f"Average Rank Position Difference (freqi): {pos_diff_freqi:.4f}")

                    print(f"Kendall Tau (frequ): {kendall_score_frequ:.4f}")
                    print(f"Spearman Rho (frequ): {spearman_score_frequ:.4f}")
                    print(f"Average Rank Position Difference (frequ): {pos_diff_frequ:.4f}")

                # if t + 1 == num_times:
                #     if not os.path.exists(plotpath + flag.rec_add):
                #         os.makedirs(plotpath + flag.rec_add)
                #     test_df_t.to_csv(plotpath + flag.rec_add + '/df_sorted.csv')
                # evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred")
                # evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred_freq")
                # evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "personal_popular")

        if flag.dataset == "ml":
            for t in [0]:
                test_df_t = test_df[test_df["idx_time"] == t]
                p_pred_test, r_pred_test = predict_batches(
                    model,
                    test_df_t["idx_user"].to_numpy(),
                    test_df_t["idx_item"].to_numpy(),
                    batch_size=5000,
                )
                p_pred_test_t = opt_scale * ((p_pred_test - np.mean(p_pred_test))/ (np.std(p_pred_test)))
                p_pred_test_t = np.clip((p_pred_test_t + opt_add), 0.0, 1.0)

                t_test_pred = np.where(p_pred_test_t >= flag.thres, 1.0, 0.0)
                p_pred_test = p_pred_test * 0.2
                r_pred_test = r_pred_test * 0.2
                test_df_t["propensity_estimate"] = np.clip(p_pred_test, 0.0001, 0.9999)
                test_df_t["relevance_estimate"] = np.clip(r_pred_test, 0.0001, 0.9999)
                outcome_estimate = test_df_t["propensity_estimate"] * test_df_t["relevance_estimate"]
                outcome_estimate = opt_scale * ((outcome_estimate - np.mean(outcome_estimate))/ (np.std(outcome_estimate)))
                outcome_estimate = np.clip((outcome_estimate + opt_add), 0.0, 1.0)
                test_df_t["outcome_estimate"] = np.where(outcome_estimate >= flag.thres, 1.0, 0.0)
                test_df_t["treated_estimate"] = t_test_pred
                causal_effect_estimate = \
                    test_df_t["outcome_estimate"] * \
                    (test_df_t["treated_estimate"] / test_df_t["propensity_estimate"] - \
                    (1 - test_df_t["treated_estimate"]) / (1 - test_df_t["propensity_estimate"]))
                test_df_t["causal_effect_estimate"] = np.clip(causal_effect_estimate, -1, 1)

                train_df = train_df[train_df.outcome>0]
                popularity = train_df["idx_item"].value_counts().reset_index()
                popularity.columns = ["idx_item", "popularity"]
                test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")

                test_df_t["pred"] = recommender.predict(test_df_t)

                evaluator = Evaluator()

                cp10_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 10))
                cp100_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CPrecS', 100))
                cdcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'CDCGS', 100000))

                cp10_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 10))
                cp100_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CPrecR', 100))
                cdcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'CDCGR', 100000))

                cp10_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 10))
                cp100_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CPrecP', 100))
                cdcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'CDCGP', 100000))

                ndcg_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'NDCGR', 10))
                ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
                ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))

                recall_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'RecallR', 10))
                recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
                recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))

                precision_tmp_list_rel.append(evaluator.evaluate(test_df_t, 'PrecisionR', 10))
                precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
                precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))

                kendall_score = evaluator.kendall_tau_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                spearman_score = evaluator.spearman_per_user(test_df_t, 'idx_user', 'pred', 'relevance_estimate')
                pos_diff = evaluator.avg_position_diff(test_df_t, 'idx_user', 'idx_item', 'pred', 'relevance_estimate')

                print(f"Kendall Tau: {kendall_score:.4f}")
                print(f"Spearman Rho: {spearman_score:.4f}")
                print(f"Average Rank Position Difference: {pos_diff:.4f}")

                evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred")
        
        if flag.dataset == "f":

            test_df_t = test_df
            train_df = train_df[train_df.outcome>0]
            popularity = train_df["idx_item"].value_counts().reset_index()
            popularity.columns = ["idx_item", "popularity"]
            test_df_t = test_df_t.merge(popularity, on="idx_item", how="left")
            test_df_t['popularity'] = (test_df_t['popularity'] - np.min(test_df_t['popularity'])) \
                                            / (np.max(test_df_t['popularity']) - np.min(test_df_t['popularity']))
            test_df_t['popularity'] = test_df_t['popularity'].fillna(0)
            test_df_t['personal_popular'] = test_df_t['personal_popular'] + test_df_t['popularity']

            test_df_t["pred"] = recommender.predict(test_df_t)
            test_df_t["pred_freq"] = recommender.predict_freq(test_df_t)
            test_df_t["pred_freqi"] = recommender.predict_freqi(test_df_t)
            test_df_t["pred_frequ"] = recommender.predict_frequ(test_df_t)

            evaluator = Evaluator()

            ndcg_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'NDCGS', 10))
            ndcg_tmp_list_pred_freq.append(evaluator.evaluate(test_df_t, 'NDCGSF', 10))
            ndcg_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_t, 'NDCGSFI', 10))
            ndcg_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_t, 'NDCGSFU', 10))
            ndcg_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'NDCGP', 10))
            ndcg_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'NDCGPP', 10))

            recall_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'RecallS', 10))
            recall_tmp_list_pred_freq.append(evaluator.evaluate(test_df_t, 'RecallSF', 10))
            recall_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_t, 'RecallSFI', 10))
            recall_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_t, 'RecallSFU', 10))
            recall_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'RecallP', 10))
            recall_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'RecallPP', 10))

            precision_tmp_list_pred.append(evaluator.evaluate(test_df_t, 'PrecisionS', 10))
            precision_tmp_list_pred_freq.append(evaluator.evaluate(test_df_t, 'PrecisionSF', 10))
            precision_tmp_list_pred_freqi.append(evaluator.evaluate(test_df_t, 'PrecisionSFI', 10))
            precision_tmp_list_pred_frequ.append(evaluator.evaluate(test_df_t, 'PrecisionSFU', 10))
            precision_tmp_list_pop.append(evaluator.evaluate(test_df_t, 'PrecisionP', 10))
            precision_tmp_list_pers_pop.append(evaluator.evaluate(test_df_t, 'PrecisionPP', 10))

            evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred")
            evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred_frequ")
            evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "pred_freqi")
            evaluator.get_dataframes(test_df_t, plotpath + flag.rec_add, "popularity")

        if flag.dataset != "f":
            cp10_pred = np.mean(cp10_tmp_list_pred)
            cp100_pred = np.mean(cp100_tmp_list_pred)
            cdcg_pred = np.mean(cdcg_tmp_list_pred)

            dirk_pred = np.mean(dirk_tmp_list_pred)
            dirk_rel = np.mean(dirk_tmp_list_rel)
            dirk_pop = np.mean(dirk_tmp_list_pop)

            dirk_coverage_pred = np.mean(dirk_coverage_tmp_list_pred)
            dirk_coverage_rel = np.mean(dirk_coverage_tmp_list_rel)
            dirk_coverage_pop = np.mean(dirk_coverage_tmp_list_pop)

            rcau_pred = np.mean(rcau_tmp_list_pred)
            rcau_rel = np.mean(rcau_tmp_list_rel)
            rcau_pop = np.mean(rcau_tmp_list_pop)

            if flag.dataset != "ml":

                dirk_pred_freq = np.mean(dirk_tmp_list_pred_freq)
                dirk_pred_freqi = np.mean(dirk_tmp_list_pred_freqi)
                dirk_pred_frequ = np.mean(dirk_tmp_list_pred_frequ)

                dirk_coverage_pred_freq = np.mean(dirk_coverage_tmp_list_pred_freq)
                dirk_coverage_pred_freqi = np.mean(dirk_coverage_tmp_list_pred_freqi)
                dirk_coverage_pred_frequ = np.mean(dirk_coverage_tmp_list_pred_frequ)

                rcau_pred_freq = np.mean(rcau_tmp_list_pred_freq)
                rcau_pred_freqi = np.mean(rcau_tmp_list_pred_freqi)
                rcau_pred_frequ = np.mean(rcau_tmp_list_pred_frequ)
                
                cp10_pred_freq = np.mean(cp10_tmp_list_pred_freq)
                cp100_pred_freq = np.mean(cp100_tmp_list_pred_freq)
                cdcg_pred_freq = np.mean(cdcg_tmp_list_pred_freq)

                cp10_pred_freqi = np.mean(cp10_tmp_list_pred_freqi)
                cp100_pred_freqi = np.mean(cp100_tmp_list_pred_freqi)
                cdcg_pred_freqi = np.mean(cdcg_tmp_list_pred_freqi)

                cp10_pred_frequ = np.mean(cp10_tmp_list_pred_frequ)
                cp100_pred_frequ = np.mean(cp100_tmp_list_pred_frequ)
                cdcg_pred_frequ = np.mean(cdcg_tmp_list_pred_frequ)

            cp10_rel = np.mean(cp10_tmp_list_rel)
            cp100_rel = np.mean(cp100_tmp_list_rel)
            cdcg_rel = np.mean(cdcg_tmp_list_rel)

            cp10_pop = np.mean(cp10_tmp_list_pop)
            cp100_pop = np.mean(cp100_tmp_list_pop)
            cdcg_pop = np.mean(cdcg_tmp_list_pop)

            if flag.dataset != "ml":
                cp10_pers_pop = np.mean(cp10_tmp_list_pers_pop)
                cp100_pers_pop = np.mean(cp100_tmp_list_pers_pop)
                cdcg_pers_pop = np.mean(cdcg_tmp_list_pers_pop)
                dirk_pers_pop = np.mean(dirk_tmp_list_pers_pop)
                dirk_coverage_pers_pop = np.mean(dirk_coverage_tmp_list_pers_pop)
                rcau_pers_pop = np.mean(rcau_tmp_list_pers_pop)
                
        if flag.dataset != "f":
            ndcg_rel = np.mean(ndcg_tmp_list_rel)
        ndcg_pred = np.mean(ndcg_tmp_list_pred)
        if flag.dataset != "ml":
            ndcg_pred_freq = np.mean(ndcg_tmp_list_pred_freq)
            ndcg_pred_freqi = np.mean(ndcg_tmp_list_pred_freqi)
            ndcg_pred_frequ = np.mean(ndcg_tmp_list_pred_frequ)
        ndcg_pop = np.mean(ndcg_tmp_list_pop)
        if flag.dataset != "ml":
            ndcg_pers_pop = np.mean(ndcg_tmp_list_pers_pop)

        if flag.dataset != "f":
            recall_rel = np.mean(recall_tmp_list_rel)
        recall_pred = np.mean(recall_tmp_list_pred)
        if flag.dataset != "ml":
            recall_pred_freq = np.mean(recall_tmp_list_pred_freq)
            recall_pred_freqi = np.mean(recall_tmp_list_pred_freqi)
            recall_pred_frequ = np.mean(recall_tmp_list_pred_frequ)
        recall_pop = np.mean(recall_tmp_list_pop)
        if flag.dataset != "ml":
            recall_pers_pop = np.mean(recall_tmp_list_pers_pop)

        if flag.dataset != "f":
            precision_rel = np.mean(precision_tmp_list_rel)
        precision_pred = np.mean(precision_tmp_list_pred)
        if flag.dataset != "ml":
            precision_pred_freq = np.mean(precision_tmp_list_pred_freq)
            precision_pred_freqi = np.mean(precision_tmp_list_pred_freqi)
            precision_pred_frequ = np.mean(precision_tmp_list_pred_frequ)
        precision_pop = np.mean(precision_tmp_list_pop)
        if flag.dataset != "ml":
            precision_pers_pop = np.mean(precision_tmp_list_pers_pop)

        if flag.dataset != "f":
            cp10list_pred.append(cp10_pred)
            cp100list_pred.append(cp100_pred)
            cdcglist_pred.append(cdcg_pred)

            if flag.dataset != "ml":
                cp10list_pred_freq.append(cp10_pred_freq)
                cp100list_pred_freq.append(cp100_pred_freq)
                cdcglist_pred_freq.append(cdcg_pred_freq)

                cp10list_pred_freqi.append(cp10_pred_freqi)
                cp100list_pred_freqi.append(cp100_pred_freqi)
                cdcglist_pred_freqi.append(cdcg_pred_freqi)
                
                cp10list_pred_frequ.append(cp10_pred_frequ)
                cp100list_pred_frequ.append(cp100_pred_frequ)
                cdcglist_pred_frequ.append(cdcg_pred_frequ)

            cp10list_rel.append(cp10_rel)
            cp100list_rel.append(cp100_rel)
            cdcglist_rel.append(cdcg_rel)

            cp10list_pop.append(cp10_pop)
            cp100list_pop.append(cp100_pop)
            cdcglist_pop.append(cdcg_pop)

            if flag.dataset != "ml":
                cp10list_pers_pop.append(cp10_pers_pop)
                cp100list_pers_pop.append(cp100_pers_pop)
                cdcglist_pers_pop.append(cdcg_pers_pop)

        # ==================== DIR@K ====================
        if flag.dataset != "f":
            dirk_list_pred.append(dirk_pred)
            
            if flag.dataset != "ml":
                dirk_list_pred_freq.append(dirk_pred_freq)
                dirk_list_pred_freqi.append(dirk_pred_freqi)
                dirk_list_pred_frequ.append(dirk_pred_frequ)
            
            dirk_list_rel.append(dirk_rel)
            dirk_list_pop.append(dirk_pop)
            
            if flag.dataset != "ml":
                dirk_list_pers_pop.append(dirk_pers_pop)

        if flag.dataset != "f":
            dirk_coverage_list_pred.append(dirk_coverage_pred)
            
            if flag.dataset != "ml":
                dirk_coverage_list_pred_freq.append(dirk_coverage_pred_freq)
                dirk_coverage_list_pred_freqi.append(dirk_coverage_pred_freqi)
                dirk_coverage_list_pred_frequ.append(dirk_coverage_pred_frequ)
            
            dirk_coverage_list_rel.append(dirk_coverage_rel)
            dirk_coverage_list_pop.append(dirk_coverage_pop)
            
            if flag.dataset != "ml":
                dirk_coverage_list_pers_pop.append(dirk_coverage_pers_pop)

        # ==================== RCAU@K ====================
        if flag.dataset != "f":
            rcau_list_pred.append(rcau_pred)
            
            if flag.dataset != "ml":
                rcau_list_pred_freq.append(rcau_pred_freq)
                rcau_list_pred_freqi.append(rcau_pred_freqi)
                rcau_list_pred_frequ.append(rcau_pred_frequ)
            
            rcau_list_rel.append(rcau_rel)
            rcau_list_pop.append(rcau_pop)
            
            if flag.dataset != "ml":
                rcau_list_pers_pop.append(rcau_pers_pop)

        if flag.dataset != "f":
            ndcglist_rel.append(ndcg_rel)
        ndcglist_pred.append(ndcg_pred)
        if flag.dataset != "ml":
            ndcglist_pred_freq.append(ndcg_pred_freq)
            ndcglist_pred_freqi.append(ndcg_pred_freqi)
            ndcglist_pred_frequ.append(ndcg_pred_frequ)
        ndcglist_pop.append(ndcg_pop)
        if flag.dataset != "ml":
            ndcglist_pers_pop.append(ndcg_pers_pop)

        if flag.dataset != "f":
            recalllist_rel.append(recall_rel)
        recalllist_pred.append(recall_pred)
        if flag.dataset != "ml":
            recalllist_pred_freq.append(recall_pred_freq)
            recalllist_pred_freqi.append(recall_pred_freqi)
            recalllist_pred_frequ.append(recall_pred_frequ)
        recalllist_pop.append(recall_pop)
        if flag.dataset != "ml":
            recalllist_pers_pop.append(recall_pers_pop)

        if flag.dataset != "f":
            precisionlist_rel.append(precision_rel)
        precisionlist_pred.append(precision_pred)
        if flag.dataset != "ml":
            precisionlist_pred_freq.append(precision_pred_freq)
            precisionlist_pred_freqi.append(precision_pred_freqi)
            precisionlist_pred_frequ.append(precision_pred_frequ)
        precisionlist_pop.append(precision_pop)
        if flag.dataset != "ml":
            precisionlist_pers_pop.append(precision_pers_pop)       

    if not os.path.isdir(plotpath):
        os.makedirs(plotpath)
    with open(plotpath + "/result_" + flag.dataset + ".txt", "a+") as f:
        if flag.dataset != "f":
            print("Models used: Propcare - ", flag.prop_type, ", Recommender - ", flag.rec_type, file=f)
            # print(f"Phi: {phi_t}, Threshold: {th}", file=f)
            # print(f"Beta used: {b}", file=f)
            print("CP10S:", np.mean(cp10list_pred), np.std(cp10list_pred), file=f)
            if flag.dataset != "ml" or flag.rec_type != "gbc":
                print("CP10SF:", np.mean(cp10list_pred_freq), np.std(cp10list_pred_freq), file=f)
                print("CP10SFI:", np.mean(cp10list_pred_freqi), np.std(cp10list_pred_freqi), file=f)
                print("CP10SFU:", np.mean(cp10list_pred_frequ), np.std(cp10list_pred_frequ), file=f)
            print("CP10R:", np.mean(cp10list_rel), np.std(cp10list_rel), file=f)
            print("CP10P:", np.mean(cp10list_pop), np.std(cp10list_pop), file=f)
            if flag.dataset != "ml":
                print("CP10PP:", np.mean(cp10list_pers_pop), np.std(cp10list_pers_pop), file=f)

            print("CP100S:", np.mean(cp100list_pred), np.std(cp100list_pred), file=f)
            if flag.dataset != "ml" or flag.rec_type != "gbc":
                print("CP100SF:", np.mean(cp100list_pred_freq), np.std(cp100list_pred_freq), file=f)
                print("CP100SFI:", np.mean(cp100list_pred_freqi), np.std(cp100list_pred_freqi), file=f)
                print("CP100SFU:", np.mean(cp100list_pred_frequ), np.std(cp100list_pred_frequ), file=f)
            print("CP100R:", np.mean(cp100list_rel), np.std(cp100list_rel), file=f)
            print("CP100P:", np.mean(cp100list_pop), np.std(cp100list_pop), file=f)
            if flag.dataset != "ml":
                print("CP100PP:", np.mean(cp100list_pers_pop), np.std(cp100list_pers_pop), file=f)
            
            print("CDCGS:", np.mean(cdcglist_pred), np.std(cdcglist_pred), file=f)
            if flag.dataset != "ml" or flag.rec_type != "gbc":
                print("CDCGSF:", np.mean(cdcglist_pred_freq), np.std(cdcglist_pred_freq), file=f)
                print("CDCGSFI:", np.mean(cdcglist_pred_freqi), np.std(cdcglist_pred_freqi), file=f)
                print("CDCGSFU:", np.mean(cdcglist_pred_frequ), np.std(cdcglist_pred_frequ), file=f)
            print("CDCGR:", np.mean(cdcglist_rel), np.std(cdcglist_rel), file=f)
            print("CDCGP:", np.mean(cdcglist_pop), np.std(cdcglist_pop), file=f)
            if flag.dataset != "ml":
                print("CDCGPP:", np.mean(cdcglist_pers_pop), np.std(cdcglist_pers_pop), file=f)

            print("DIR10S:", np.mean(dirk_list_pred), np.std(dirk_list_pred), 
                    " with coverage: ", np.mean(dirk_coverage_list_pred), np.std(dirk_coverage_list_pred), file=f)
            if flag.dataset != "ml" or flag.rec_type != "gbc":
                print("DIR10SF:", np.mean(dirk_list_pred_freq), np.std(dirk_list_pred_freq),
                        " with coverage: ", np.mean(dirk_coverage_list_pred_freq), np.std(dirk_coverage_list_pred_freq), file=f)
                print("DIR10SFI:", np.mean(dirk_list_pred_freqi), np.std(dirk_list_pred_freqi),
                        " with coverage: ", np.mean(dirk_coverage_list_pred_freqi), np.std(dirk_coverage_list_pred_freqi), file=f)
                print("DIR10SFU:", np.mean(dirk_list_pred_frequ), np.std(dirk_list_pred_frequ),
                        " with coverage: ", np.mean(dirk_coverage_list_pred_frequ), np.std(dirk_coverage_list_pred_frequ), file=f)
            print("DIR10R:", np.mean(dirk_list_rel), np.std(dirk_list_rel), 
                    " with coverage: ", np.mean(dirk_coverage_list_rel), np.std(dirk_coverage_list_rel),file=f)
            print("DIR10P:", np.mean(dirk_list_pop), np.std(dirk_list_pop), 
                    " with coverage: ", np.mean(dirk_coverage_list_pop), np.std(dirk_coverage_list_pop),file=f)
            if flag.dataset != "ml":
                print("DIR10PP:", np.mean(dirk_list_pers_pop), np.std(dirk_list_pers_pop), 
                        " with coverage: ", np.mean(dirk_coverage_list_pers_pop), np.std(dirk_coverage_list_pers_pop),file=f)

            print("RCAU10S:", np.mean(rcau_list_pred), np.std(rcau_list_pred), file=f)
            if flag.dataset != "ml" or flag.rec_type != "gbc":
                print("RCAU10SF:", np.mean(rcau_list_pred_freq), np.std(rcau_list_pred_freq), file=f)
                print("RCAU10FI:", np.mean(rcau_list_pred_freqi), np.std(rcau_list_pred_freqi), file=f)
                print("RCAU10SFU:", np.mean(rcau_list_pred_frequ), np.std(rcau_list_pred_frequ), file=f)
            print("RCAU10R:", np.mean(rcau_list_rel), np.std(rcau_list_rel), file=f)
            print("RCAU10P:", np.mean(rcau_list_pop), np.std(rcau_list_pop), file=f)
            if flag.dataset != "ml":
                print("RCAU10PP:", np.mean(rcau_list_pers_pop), np.std(rcau_list_pers_pop), file=f)

        print("NDCG10S:", np.mean(ndcglist_pred), np.std(ndcglist_pred), file=f)
        if flag.dataset != "ml" or flag.rec_type != "gbc":
            print("NDCG10SF:", np.mean(ndcglist_pred_freq), np.std(ndcglist_pred_freq), file=f)
            print("NDCG10SFI:", np.mean(ndcglist_pred_freqi), np.std(ndcglist_pred_freqi), file=f)
            print("NDCG10SFU:", np.mean(ndcglist_pred_frequ), np.std(ndcglist_pred_frequ), file=f)
        if flag.dataset != "f":
            print("NDCG10R:", np.mean(ndcglist_rel), np.std(ndcglist_rel), file=f)
        print("NDCG10P:", np.mean(ndcglist_pop), np.std(ndcglist_pop), file=f)
        if flag.dataset != "ml":
            print("NDCG10PP:", np.mean(ndcglist_pers_pop), np.std(ndcglist_pers_pop), file=f)

        print("Recall10S:", np.mean(recalllist_pred), np.std(recalllist_pred), file=f)
        if flag.dataset != "ml" or flag.rec_type != "gbc":
            print("Recall10SF:", np.mean(recalllist_pred_freq), np.std(recalllist_pred_freq), file=f)
            print("Recall10SFI:", np.mean(recalllist_pred_freqi), np.std(recalllist_pred_freqi), file=f)
            print("Recall10SFU:", np.mean(recalllist_pred_frequ), np.std(recalllist_pred_frequ), file=f)
        if flag.dataset != "f":
            print("Recall10R:", np.mean(recalllist_rel), np.std(recalllist_rel), file=f)
        print("Recall10P:", np.mean(recalllist_pop), np.std(recalllist_pop), file=f)
        if flag.dataset != "ml":
            print("Recall10PP:", np.mean(recalllist_pers_pop), np.std(recalllist_pers_pop), file=f)

        print("Precision10S:", np.mean(precisionlist_pred), np.std(precisionlist_pred), file=f)
        if flag.dataset != "ml" or flag.rec_type != "gbc":
            print("Precision10SF:", np.mean(precisionlist_pred_freq), np.std(precisionlist_pred_freq), file=f)
            print("Precision10SFI:", np.mean(precisionlist_pred_freqi), np.std(precisionlist_pred_freqi), file=f)
            print("Precision10SFU:", np.mean(precisionlist_pred_frequ), np.std(precisionlist_pred_frequ), file=f)
        if flag.dataset != "f":
            print("Precision10R:", np.mean(precisionlist_rel), np.std(precisionlist_rel), file=f)
        print("Precision10P:", np.mean(precisionlist_pop), np.std(precisionlist_pop), file=f) 
        if flag.dataset != "ml":
            print("Precision10PP:", np.mean(precisionlist_pers_pop), np.std(precisionlist_pers_pop), file=f) 
        print("--------------------------------", file=f)    
            
if __name__ == "__main__":
    physical_devices = tf.config.list_physical_devices('MPS')
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass
    main(flag)