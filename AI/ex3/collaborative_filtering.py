# Noam Koren
# 308192871
import sys
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
        """
        userId is 283238
        mostly likes Action | SciFi
        """
        id = 283238
        ur = [{'userId': id, 'movieId': 59315, 'rating': 5.0}, {'userId': id, 'movieId': 51662, 'rating': 5.0}, {'userId': id, 'movieId': 56174, 'rating': 3.5}, {'userId': id, 'movieId': 68319, 'rating': 4.0}, {'userId': id, 'movieId': 86332, 'rating': 4.0}, {'userId': id, 'movieId': 89745, 'rating': 4.5}, {'userId': id, 'movieId': 72998, 'rating': 5}, {'userId': id, 'movieId': 6534, 'rating': 4.0}, {'userId': id, 'movieId': 5349, 'rating': 4.5}]
        rating = rating.append(ur, ignore_index = True)
        return rating

    def create_movie_dictionary(self,movies):
        for  id, title in zip(movies.movieId, movies.title):
            self.movie_dict[id] = title

    def create_user_based_matrix(self, data):
        self.data = data
        _ratings, movies = data
        self.create_movie_dictionary(movies)
        self.data_matrix = _ratings.pivot(index='userId', columns='movieId', values='rating')
        self.data_matrix.rename(columns=self.movie_dict, inplace=True)
        ratings = self.data_matrix.to_numpy()
        mean_user_rating = self.data_matrix.mean(axis=1).to_numpy().reshape(-1, 1).round(2)
        ratings_diff = ratings - mean_user_rating
        ratings_diff[np.isnan(ratings_diff)]=0
        ratings_diff = ratings_diff.round(2)
        
        user_similarity = 1-pairwise_distances(ratings_diff, metric='cosine')
        pred = mean_user_rating + user_similarity.dot(ratings_diff) / np.array([np.abs(user_similarity).sum(axis=1)]).T
        self.user_based_matrix = pd.DataFrame(pred, columns=self.data_matrix.columns, index=self.data_matrix.index)

    def create_item_based_matrix(self, data):
        self.data = data
        _ratings, movies = data
        self.create_movie_dictionary(movies)
        self.data_matrix = _ratings.pivot(index='userId', columns='movieId', values='rating')
        self.data_matrix.rename(columns=self.movie_dict, inplace=True)
        ratings = self.data_matrix.to_numpy()
        mean_user_rating = self.data_matrix.mean(axis=1).to_numpy().reshape(-1, 1).round(2)
        ratings_diff = ratings - mean_user_rating
        ratings_diff[np.isnan(ratings_diff)]=0
        ratings_diff = ratings_diff.round(2)
        
        item_similarity = 1-pairwise_distances(ratings_diff.T, metric='cosine')
        pred = mean_user_rating.T + item_similarity.dot(ratings_diff.T) / np.array([np.abs(item_similarity).sum(axis=1)]).T
        self.item_based_metrix = pd.DataFrame(pred, columns=self.data_matrix.index, index=self.data_matrix.columns)

    def predict_movies(self, user_id, k, is_user_based=True):
        if is_user_based:
            data_row_unrated_pred = self.user_based_matrix.loc[int(user_id)][np.isnan(self.data_matrix.loc[int(user_id)])]
        else:
            data_row_unrated_pred = self.item_based_metrix[int(user_id)][np.isnan(self.data_matrix.loc[int(user_id)])]
        idx = np.argsort(-data_row_unrated_pred)
        sim_scores = idx[0:k]
        return data_row_unrated_pred.iloc[sim_scores]
