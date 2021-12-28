import sys
from matplotlib import colors
import matplotlib.pyplot as plt
import pandas as pd
from pandas.core.algorithms import unique
from collections import Counter
import seaborn as sns
import numpy as np

def watch_data_info(data):
    for d in data:
        # This function returns the first 5 rows for the object based on position.
        # It is useful for quickly testing if your object has the right type of data in it.
        print(d.head())

        # This method prints information about a DataFrame including the index dtype and column dtypes, non-null values and memory usage.
        print(d.info())

        # Descriptive statistics include those that summarize the central tendency, dispersion and shape of a dataset’s distribution, excluding NaN values.
        print(d.describe(include='all').transpose())


def print_data(data):
    ratings, movies = data
    # Question1:
    total_ratings_count = ratings.userId.count()
    unique_user_rating_count = ratings.userId.drop_duplicates().count()
    unique_movies_rated_count = movies.movieId.drop_duplicates().count()
    print(f"total ratings: {total_ratings_count}\ntotal users rated: {unique_user_rating_count}\ntotal movies rated: {unique_movies_rated_count}")

    # rating distribution
    # c = Counter(ratings.rating).most_common(15)
    # y = []
    # x = []
    # for r, count in c:
    #     x.append(r)
    #     y.append(count)
    # c = ['#581845', '#900C3F', '#C70039', '#FF5733', '#FFC300', '#DAF7A6', '#33acff']
    # plt.bar(x,y, color=c)
    # plt.show()

    # # Quiestion 2:
    # movie_dict = {}
    # movie_count_dict = {}
    # for movie in movies.values:
    #     movie_dict[movie[1]] = movie[2]
    # c = Counter(ratings.movieId).most_common(5)
    # for id, count in c:
    #     movie_count_dict[movie_dict[id]] = count
    # plt.ylabel = "ratings"
    # plt.bar(movie_count_dict.keys(), movie_count_dict.values())
    # plt.show()
    # print(c)
    # plot_data(ratings)
    # watch_data_info(ratings)
    sys.exit(1)

def plot_data(data, plot = True):
    plt.plot(data)
    if plot:
        plt.show()
    sys.exit(1)
