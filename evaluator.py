import numpy as np
from scipy.stats import kendalltau
from scipy.stats import spearmanr

class Evaluator():
    def __init__(self,
                 colname_user='idx_user', colname_item='idx_item', colname_time='idx_time',
                 colname_outcome='outcome', colname_prediction='pred', colname_prediction_freq='pred_freq',
                 colname_prediction_freqi='pred_freqi', colname_prediction_frequ='pred_frequ',
                 colname_treatment='treated', colname_propensity='propensity',
                 colname_effect='causal_effect', colname_estimate='causal_effect_estimate',
                 colname_relavance = 'relevance_estimate', colname_popularity = 'popularity', 
                 colname_personal_popularity = 'personal_popular'):


        self.rank_k = None
        self.colname_user = colname_user
        self.colname_item = colname_item
        self.colname_time = colname_time
        self.colname_outcome = colname_outcome
        self.colname_relavance = colname_relavance
        self.colname_prediction = colname_prediction
        self.colname_prediction_freq = colname_prediction_freq
        self.colname_prediction_freqi = colname_prediction_freqi
        self.colname_prediction_frequ = colname_prediction_frequ
        self.colname_treatment = colname_treatment
        self.colname_propensity = colname_propensity
        self.colname_effect = colname_effect
        self.colname_estimate = colname_estimate
        self.colname_popularity = colname_popularity
        self.colname_personal_popularity = colname_personal_popularity

    def get_ranking(self, df, sort_by = 'pred', num_rec=10):
        df = df.sort_values(by=[self.colname_user, sort_by], ascending=False)
        df_ranking = df.groupby(self.colname_user).head(num_rec)
        return df_ranking

    def get_sorted(self, df, sort_by = 'pred'):
        df = df.sort_values(by=[self.colname_user, sort_by], ascending=False)
        return df
    
    def get_dataframes(self, df, path, sort_by):
        df = df.sort_values(by=[self.colname_user, sort_by], ascending=False)
        df_ranking_10 = df.groupby(self.colname_user).head(10)
        df_ranking_100 = df.groupby(self.colname_user).head(100)

        df_ranking_10.to_csv(path + sort_by + '/df_ranking_10.csv')
        df_ranking_100.to_csv(path + sort_by + '/df_ranking_100.csv') 

    def capping(self, df, cap_prop=None):
        if cap_prop is not None and cap_prop > 0:
            bool_cap = np.logical_and(df.loc[:, self.colname_propensity] < cap_prop,
                                      df.loc[:, self.colname_treatment] == 1)
            if np.sum(bool_cap) > 0:
                df.loc[bool_cap, self.colname_propensity] = cap_prop

            bool_cap = np.logical_and(df.loc[:, self.colname_propensity] > 1 - cap_prop,
                                      df.loc[:, self.colname_treatment] == 0)
            if np.sum(bool_cap) > 0:
                df.loc[bool_cap, self.colname_propensity] = 1 - cap_prop

        return df

    def clip(self, df, cap_prop=None):
        if cap_prop is not None and cap_prop > 0:
            pvalue = df[self.colname_propensity].values
            pvalue = np.clip(pvalue, cap_prop, 1-cap_prop)
            df[self.colname_propensity] = pvalue
        return df
    
    def kendall_tau_per_user(self, df, user_col, rank_col_1, rank_col_2):
        taus = []
        for _, group in df.groupby(user_col):
            if len(group) > 1:
                order_1 = group[rank_col_1].rank(ascending=False, method='first')
                order_2 = group[rank_col_2].rank(ascending=False, method='first')
                tau, _ = kendalltau(order_1, order_2)
                taus.append(tau)
        return np.nanmean(taus)
    

    def spearman_per_user(self,df, user_col, rank_col_1, rank_col_2):
        rhos = []
        for _, group in df.groupby(user_col):
            if len(group) > 1:
                order_1 = group[rank_col_1].rank(ascending=False, method='first')
                order_2 = group[rank_col_2].rank(ascending=False, method='first')
                rho, _ = spearmanr(order_1, order_2)
                rhos.append(rho)
        return np.nanmean(rhos)
    
    def avg_position_diff(self,df, user_col, item_col, pred_col, rel_col):
        diffs = []
        for _, group in df.groupby(user_col):
            group = group.copy()
            group['rank_pred'] = group[pred_col].rank(ascending=False, method='first')
            group['rank_rel'] = group[rel_col].rank(ascending=False, method='first')
            diffs.extend(np.abs(group['rank_pred'] - group['rank_rel']))
        return np.nanmean(diffs)


    def evaluate(self, df_origin, measure, num_rec, mode = 'ASIS', cap_prop=0.0):
        df_sorted = df_origin.copy(deep=True)
        # print(df.head())
        df_sorted = self.capping(df_sorted, cap_prop)
        # df = self.clip(df, cap_prop)
        # print(df.head())
        self.rank_k = num_rec

        if 'IPS' in measure:
            df_sorted.loc[:, self.colname_estimate] = df_sorted.loc[:, self.colname_outcome] * \
                                        (df_sorted.loc[:, self.colname_treatment] / df_sorted.loc[:,self.colname_propensity] - \
                                         (1 - df_sorted.loc[:, self.colname_treatment]) / (1 - df_sorted.loc[:, self.colname_propensity]))
        if measure == 'precision':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_outcome: self.prec_at_k})))

        elif measure == 'DIR@KS':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KSF':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KSFI':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KSFU':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KR':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KP':
            return self._compute_dir_at_k(df_sorted, num_rec)
        elif measure == 'DIR@KPP':
            return self._compute_dir_at_k(df_sorted, num_rec)

        elif measure == 'RCAU@KS':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KSF':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KSFI':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KSFU':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KR':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KP':
            return self._compute_rcau_at_k(df_sorted, num_rec)
        elif measure == 'RCAU@KPP':
            return self._compute_rcau_at_k(df_sorted, num_rec)

        elif measure == 'Prec':
            df_ranking = self.get_ranking(df_sorted, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_outcome].values)
        elif measure == 'CPrecS':
            df_ranking = self.get_ranking(df_sorted, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecSF':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_prediction_freq, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecSFI':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_prediction_freqi, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecSFU':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_prediction_frequ, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecR':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_relavance, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values)
        elif measure == 'CPrecP':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_popularity, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values) 
        elif measure == 'CPrecPP':
            df_ranking = self.get_ranking(df_sorted, sort_by=self.colname_personal_popularity, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_effect].values) 
        elif measure == 'CPrecIPS':
            df_ranking = self.get_ranking(df_sorted, num_rec=num_rec)
            return np.nanmean(df_ranking.loc[:, self.colname_estimate].values)
        elif measure == 'DCG':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_outcome: self.dcg_at_k})))
        elif measure == 'CDCGS':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGSF':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGSFI':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGSFU':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGR':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGP':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGPP':
            return float(np.nanmean(df_sorted.groupby(self.colname_user)[self.colname_effect]
                                    .apply(lambda x: self.dcg_at_k(x))))
        elif measure == 'CDCGIPS':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_estimate: self.dcg_at_k})))
        elif measure == 'AR':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_outcome: self.ave_rank})))
        elif measure == 'CAR':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.ave_rank})))
        elif measure == 'CARIPS':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_estimate: self.ave_rank})))
        elif measure == 'CARP':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.arp})))
        elif measure == 'CARPIPS':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_estimate: self.arp})))
        elif measure == 'CARN':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.arn})))
        elif measure == 'CARNIPS':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_estimate: self.arn})))
        elif measure == 'hit':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_outcome: self.hit_at_k})))
        elif measure == 'AUC':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_outcome: self.auc})))
        elif measure == 'CAUC':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.gauc})))
        elif measure == 'CAUCP':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.gaucp})))
        elif measure == 'CAUCN':
            return float(np.nanmean(df_sorted.groupby(self.colname_user).agg({self.colname_effect: self.gaucn})))
        elif measure == 'RecallS':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_prediction)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallSF':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_prediction_freq)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallSFI':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_prediction_freqi)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallSFU':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_prediction_frequ)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallR':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_relavance)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallP':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_popularity)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'RecallPP':
            recall_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.recall_at_k(x, sort_by=self.colname_personal_popularity)
            )
            return float(np.nanmean(recall_scores))
        elif measure == 'PrecisionS':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_prediction)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionSF':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_prediction_freq)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionSFI':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_prediction_freqi)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionSFU':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_prediction_frequ)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionR':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_relavance)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionP':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_popularity)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'PrecisionPP':
            precision_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.precision_at_k(x, sort_by=self.colname_personal_popularity)
            )
            return float(np.nanmean(precision_scores))
        elif measure == 'NDCGS':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_prediction, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGSF':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_prediction_freq, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGSFI':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_prediction_freqi, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGSFU':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_prediction_frequ, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGR':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_relavance, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGP':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_popularity, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        elif measure == 'NDCGPP':
            ndcg_scores = df_sorted.groupby(self.colname_user).apply(
                lambda x: self.ndcg_at_k(x, sort_by=self.colname_personal_popularity, label_col=self.colname_outcome)
            )
            return float(np.nanmean(ndcg_scores))
        else:
            print('measure:"' + measure + '" is not supported! ')

    def _compute_dir_at_k(self, df, k, min_items=3, re_sort='no'):
        df['propensity_estimate'] = df['propensity_estimate'].clip(0.01, 0.99)  # Avoid extreme propensities

        df['causal_effect_estimate'] = df['outcome'] * \
            (df['treated'] / df['propensity_estimate'] - \
                (1 - df['treated']) / (1 - df['propensity_estimate']))
        
        valid_effects = []
        
        for _, group in df.groupby('idx_user'):
            if len(group) >= min_items:
                if re_sort == 'max':
                    # For theoretical bounds
                    group_sorted = group.sort_values(by='causal_effect_estimate', ascending=False)
                elif re_sort == 'min':
                    group_sorted = group.sort_values(by='causal_effect_estimate', ascending=True)
                else:
                    # For actual ranking evaluation
                    group_sorted = group
                
                top_k = group_sorted.head(k)
                valid_effects.append(top_k['causal_effect_estimate'].mean())
        
        total_users = df['idx_user'].nunique()
        coverage = len(valid_effects) / total_users if total_users > 0 else 0
        metric = np.mean(valid_effects) if valid_effects else np.nan
        
        return metric, coverage

    def _compute_rcau_at_k(self, df, k):
        def rcau_per_user(group):
            n_items = len(group)
            actual_k = min(k, n_items)
            
            if actual_k == 0:
                return 0, 0
            
            top_k = group.head(actual_k)
            avg_uplift = top_k['causal_effect'].mean()
            
            # Completeness penalty: (actual_k / k)^2
            completeness_penalty = (actual_k / k) ** 2
            
            return avg_uplift * completeness_penalty, completeness_penalty
        
        weighted_sum = 0
        total_weight = 0
        
        for _, group in df.groupby('idx_user'):
            val, w = rcau_per_user(group)
            weighted_sum += val
            total_weight += w
        
        return weighted_sum / total_weight if total_weight > 0 else np.nan

    # functions for each metric
    def prec_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return sum(x[:k]) / k

    def dcg_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return np.sum(x[:k] / np.log2(np.arange(k) + 2))

    def ndcg_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        max_dcg_at_k = self.dcg_at_k(sorted(x, reverse=True))
        if max_dcg_at_k == 0:
            return np.nan
        else:
            return self.dcg_at_k(x) / max_dcg_at_k

    def hit_at_k(self, x):
        k = min(self.rank_k, len(x))  # rank_k is global variable
        return float(any(x[:k] > 0))

    def auc(self, x): # for binary (1/0)
        len_x = len(x)
        idx_posi = np.where(x > 0)[0]
        len_posi = len(idx_posi)
        len_nega = len_x - len_posi
        if len_posi == 0 or len_nega == 0:
            return np.nan
        cnt_posi_before_posi = (len_posi * (len_posi - 1)) / 2
        cnt_nega_before_posi = np.sum(idx_posi) - cnt_posi_before_posi
        return 1 - cnt_nega_before_posi / (len_posi * len_nega)

    def gauc(self, x): # AUC with ternary (1/0/-1) value
        x_p = x > 0
        x_n = x < 0
        num_p = np.sum(x_p)
        num_n = np.sum(x_n)
        gauc = 0.0
        if num_p > 0:
            gauc += self.auc(x_p) * (num_p/(num_p + num_n))
        if num_n > 0:
            gauc += (1.0 - self.auc(x_n)) * (num_n/(num_p + num_n))
        return gauc

    def gaucp(self, x):
        return self.auc(x > 0)

    def gaucn(self, x):
        return self.auc(x < 0)

    def ave_rank(self, x):
        len_x = len(x)
        rank = np.arange(len_x) + 1
        return np.mean(x * rank)

    def arp(self, x):
        return self.ave_rank(x > 0)
    def arn(self, x):
        return self.ave_rank(x < 0)

    def dcg_at_k(self, rel_values):
        k = min(self.rank_k, len(rel_values))
        return np.sum(rel_values[:k] / np.log2(np.arange(k) + 2))

    def ndcg_at_k(self, df_user, sort_by, label_col='outcome'):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        rel = df_user[label_col].values

        k = min(self.rank_k, len(rel))
        ideal_rel = np.sort(rel)[::-1]  # for ideal ranking

        dcg = self.dcg_at_k(rel)
        idcg = self.dcg_at_k(ideal_rel)

        return dcg / idcg if idcg > 0 else np.nan
    
    def recall_at_k(self, df_user, sort_by):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        k = min(self.rank_k, len(df_user))
        rel_in_top_k = df_user['outcome'].iloc[:k].sum()
        total_rel = df_user['outcome'].sum()

        return rel_in_top_k / total_rel if total_rel > 0 else np.nan
    
    def precision_at_k(self, df_user, sort_by):
        df_user = df_user.sort_values(by=sort_by, ascending=False)
        k = min(self.rank_k, len(df_user))
        rel_in_top_k = df_user['outcome'].iloc[:k].sum()
        
        return rel_in_top_k / k if k > 0 else np.nan