import sys
import pandas as pd
import numpy as np
import heapq
from sklearn.metrics.pairwise import pairwise_distances


class collaborative_filtering:
    def __init__(self):
        self.user_based_matrix = []
        self.item_based_metrix = []

    def create_fake_user(self,rating):
        "*** YOUR CODE HERE ***"
        return rating


    def create_user_based_matrix(self, data):
        ratings = data[0]

        #for adding fake user
        ratings = self.create_fake_user(ratings)

        "*** YOUR CODE HERE ***"
        sys.exit(1)

    def create_item_based_matrix(self, data):
        "*** YOUR CODE HERE ***"
        sys.exit(1)

    def predict_movies(self, user_id, k, is_user_based=True):
        "*** YOUR CODE HERE ***"
        sys.exit(1)
