import sys
from numpy.core.numeric import moveaxis
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []
        self.data_matrix = []
        self.movie_dict = {}

    def create_fake_user(self, rating):
        "*** YOUR   CODE HERE ***"
                                                                                        
        return rating

    def create_user_based_matrix(self, data):
        self.data = data
        _ratings, movies = data
        self.data_matrix = _ratings.pivot(index='userId', columns='movieId', values='rating')
        ratings = self.data_matrix.to_numpy()
        mean_user_rating = self.data_matrix.mean(axis=1).to_numpy().reshape(-1, 1).round(2)
        ratings_diff = ratings - mean_user_rating
        ratings_diff[np.isnan(ratings_diff)]=0
        ratings_diff = ratings_diff.round(2)
        
        user_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        pred.round(2)
        self.user_based_matrix = pd.DataFrame(pred, columns=self.data_matrix.columns, index=self.data_matrix.index)

    def create_item_based_matrix(self, data):
        self.data = data
        r, m = data
        self.data_matrix = self.create_pref_matrix(data)
        m = self.data_matrix
        ratings = m.to_numpy()
        mean_user_rating = m.mean(axis=1).to_numpy().reshape(-1, 1)
        mean_user_rating.round(2)

        ratings_diff = (ratings - mean_user_rating)
        ratings_diff[np.isnan(ratings_diff)]=0
        ratings_diff.round(2)

        raitingItem = ratings_diff
        raitingItem[np.isnan(raitingItem)]=0
        item_similarity = 1-pairwise_distances(raitingItem.T, metric='cosine')
        pred = mean_user_rating + raitingItem.dot(item_similarity) / np.array([np.abs(item_similarity).sum(axis=1)])
        pred.round(2)
        self.item_based_metrix = pd.DataFrame(pred, columns=self.data_matrix.columns, index=self.data_matrix.index)

    def predict_movies(self, user_id, k, is_user_based=True):
        if is_user_based:
            data_row_unrated_pred = self.user_based_matrix.loc[int(user_id)][np.isnan(self.data_matrix.loc[int(user_id)])]
        else:
            data_row_unrated_pred = self.item_based_metrix.loc[int(user_id)][np.isnan(self.data_matrix.loc[int(user_id)])]
        idx = np.argsort(-data_row_unrated_pred)
        sim_scores = idx[0:k]
        return data_row_unrated_pred.iloc[sim_scores]

        sys.exit(1)
