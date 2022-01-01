# Noam Koren
# 308192871
from sklearn.metrics import mean_squared_error
from math import sqrt
# Import Pandas
import pandas as pd

def precision_10(test_set, cf, is_user_based = True):
    vals = 0
    m = test_set.pivot(columns='movieId', index='userId', values='rating')
    m.rename(columns=cf.movie_dict, inplace = True)
    for user in m.index:
        pred = cf.predict_movies(user, 10, is_user_based)
        row = m.loc[user][pred.index]
        val = 0
        for i in range(10):
            if row.values[i] >= 4:
                val +=1
        vals += val / 10
    vals /= len(m.index.values)
    print("Precision_k: " + str(vals))

def ARHA(test_set, cf, is_user_based = True):
    m = test_set.pivot(columns='movieId', index='userId', values='rating')
    m.rename(columns=cf.movie_dict, inplace = True)
    val = 0
    for user in m.index:
        pred = cf.predict_movies(user, 10, is_user_based)
        row = m.loc[user][pred.index]
        temp = 0
        for i in range(pred.size):
            if row.values[i] >= 4.0: 
                temp += 1/(i+1)
        val += temp
    val /= len(m.index.values)
    print("ARHR: " + str(val))

def RSME(test_set, cf, is_user_based = True):
    m = test_set.pivot(columns='movieId', index='userId', values='rating')
    m.rename(columns=cf.movie_dict, inplace = True)
    val = 0
    for user in m.index:
        pred = cf.predict_movies(user, 10, is_user_based)
        user_row = m.loc[user][pred.index].dropna()
        pred = pred[user_row.index]
        if pred.values.size > 0 and user_row.values.size > 0:
            val += mean_squared_error(user_row.values, pred.values)
    val /= m.index.size
    print("RMSE: " + str(val))