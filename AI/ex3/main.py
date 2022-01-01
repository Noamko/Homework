import data as da
import collaborative_filtering as cf
import evaluation as ev
# Import Pandas
import pandas as pd

# Load restaurant data
movies = pd.read_csv('data/movies_subset.csv', low_memory=False)
# Load rating data
rating = pd.read_csv('data/ratings.csv', low_memory=False)
# Load test rating data
test_set = pd.read_csv('data/test.csv', low_memory=False)

cf1 = cf.collaborative_filtering()

# PART 1 - DATA
def analsys(data):
    # da.watch_data_info(data)
    da.print_data(data)
    da.plot_data(data, True)

# PATR 2 - COLLABORATING FILLTERING RECOMMENDATION SYSTEM
def collaborative_filtering_rec(data, user_based = True):
    global cf1
    if(user_based):
        cf1.create_user_based_matrix(data)
    else:
        cf1.create_item_based_matrix(data)

    

# PART 3 - EVALUATION
def evaluate_rec():
    ev.precision_10(test_set, cf1,False)
    ev.ARHA(test_set,cf1, False)
    ev.RSME(test_set, cf1, False)

def main():
    #data
    # r = cf1.create_fake_user(rating)
    # analsys((rating, movies))

    # #collaborative filtering
    collaborative_filtering_rec((rating, movies), False)
    result = cf1.predict_movies(283225, 5, False)
    print(result)


    # #evaluation
    evaluate_rec()


if __name__ == "__main__":
    main()







