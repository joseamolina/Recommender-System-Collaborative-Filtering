"""
@Author:    Jose Angel Molina
@Date:      Nov 2018
@Company:   CIT
"""
import pandas as pd
import numpy as np
import random
import sklearn.metrics.pairwise as sc2

class RecCFSys:

    """
    Constructor: It assign hyper parameters to the class, reads the dataset and assign the seed
    """
    def __init__(self, filename: 'String'):

        self.k = None
        self.v = None
        self.target_id = None

        # (Under prediction, over prediction)
        # Seed the number of Jose: R00156440
        random.seed(156440)

        self.conf_matrix = [[0,0], [0, 0]]

        # Read file
        self.data_set_films = np.genfromtxt(filename, int)

    """
    It generates a prediction
    """
    def make_prediction(self, test_per, mean_centering, sim_func, k):

        self.conf_matrix = [[0,0], [0, 0]]
        self.bias = (0, 0)

        # Get all users
        list_things = np.unique(self.data_set_films[:, 0])

        # Percentage of testing users
        num = round(test_per * len(list_things) / 100)

        # Get the training data set
        training_dataset = list_things[num:]

        # Get the testing dataset
        testing_dataset = list_things[:num]

        # Series of all users
        users_to_predict = pd.Series(testing_dataset)

        # Apply a prediction and return the accuracy for each users od the testing dataset
        users_to_predict = users_to_predict.apply(lambda t: self.predict_obt(t, training_dataset, test_per, mean_centering, sim_func, k))

        print(
            'Prediction for a testing percentage of {0}%, K = {1}, a similarity function using {2} and using {3}'.format(
                test_per, k, mean_centering, sim_func))

        # Print accuracy of the recommender system
        print('The Recall of the method for like predictions is {0}'.format(self.conf_matrix[0][0] / (self.conf_matrix[0][0] + self.conf_matrix[1][0])))
        print('The Precission of the method for like predictions is {0}'.format(
            self.conf_matrix[0][0] / (self.conf_matrix[0][0] + self.conf_matrix[0][1])))

        pred = 'under predictive' if self.bias[0] > self.bias[1] else 'over predictive'

        und_pred = (self.bias[0] * 100) / (self.bias[0] + self.bias[1])
        print('The recommender system tends to be {0}, with a percentage of {1}% for under prediction and {2}% for over prediction'.format(pred, round(und_pred, 2), round(100-und_pred, 2)))

        abs_mean = 0
        sqr_mean = 0
        spearman = 0
        it_nnull = 0
        for seq in users_to_predict:
            if seq is not np.nan:
                it_nnull += 1
                abs_mean += seq[0]
                sqr_mean += seq[1]
                spearman += seq[2]

        abs_mean /= it_nnull
        sqr_mean /= it_nnull
        spearman /= it_nnull

        print('The Mean absolute error is {0}'.format(abs_mean))
        print('The Mean Squared Error is {0}'.format(np.sqrt(sqr_mean)))
        print('The general Spearman rank correlation is {0}'.format(spearman))
        print('\n\n')

    def predict_obt(self, user, training, tes_perc, mean_centering, sim_func, k):

        # Get all items of a given user
        rates_user_bool = self.data_set_films[:, 0] == user
        items_user = self.data_set_films[rates_user_bool][:, 1]

        # Calculate the percentage of testing items to predict
        perc = round(tes_perc * len(items_user) / 100)

        # Get items to be predicted randomly
        testing_items = np.random.choice(items_user, perc)

        if len(testing_items) != 0:

            # Items to be compared
            training_items_a = list(set(items_user) - set(testing_items))

            # Dictionary containing the distance for each user-neighbour
            distance = {}

            # Dictionary containing the appropiate consolidation for values of the neighbour
            comps = {}

            # For all neighbours
            for neig in training:

                # Get all items of a neighbour
                rates_neig = self.data_set_films[:, 0] == neig
                rates_user_neig = self.data_set_films[rates_neig][:, 1]

                # Check if the testing items are contained in the neighbour
                if all(np.isin(testing_items, rates_user_neig)):

                    # Intersect the rest of the training items of both user and neighbour
                    common_films = np.intersect1d(training_items_a, rates_user_neig)

                    # Check if both they have common training items rated
                    if len(common_films) != 0:

                        # For both user and neighbour, get the rates of common items in order.
                        rates_a = []
                        rates_b = []

                        for item in common_films:
                            rat_A = self.data_set_films[
                                (self.data_set_films[:, 1] == item) & (self.data_set_films[:, 0] == user)]

                            rat_B = self.data_set_films[
                                (self.data_set_films[:, 1] == item) & (self.data_set_films[:, 0] == neig)]

                            rates_a.append(int(rat_A[:, 2]))
                            rates_b.append(int(rat_B[:, 2]))

                        # Calculate the similarity function
                        if sim_func == 'adj-cosine':
                            distance[neig] = (
                                sc2.cosine_similarity(np.array(rates_a).reshape(1, -1),
                                                      np.array(rates_b).reshape(1, -1)))
                        else:
                            avg_a = np.mean(rates_a)
                            avg_b = np.mean(rates_b)
                            distance[neig] = (np.sum((rates_a - avg_a) * (rates_b - avg_b)) / (
                                    np.sqrt(np.sum(np.square(rates_a - avg_a))) * np.sqrt(
                                np.sum(np.square(rates_b - avg_b)))))

                        # Doing the either mean centering or the z-scores for normalization
                        # in corresponding testing items in neighbour
                        ref = []
                        for pel in testing_items:
                            rates_user_neig = self.data_set_films[
                                                  (self.data_set_films[:, 0] == neig) & (self.data_set_films[:,
                                                                                         1] == pel)][:, 2]
                            if mean_centering:
                                ref.append(rates_user_neig - np.mean(rates_b))
                            else:
                                ref.append((rates_user_neig - np.mean(rates_b)) / np.std(rates_b))

                        comps[neig] = ref

            # Check if the candidate neighbours are less than k
            if len(distance) >= k:

                all_rates = []
                testing_real_rates = []
                for item in items_user:
                    rat_A = self.data_set_films[
                        (self.data_set_films[:, 1] == item) & (self.data_set_films[:, 0] == user)]

                    all_rates.append(int(rat_A[:, 2]))

                for item in testing_items:
                    rat_A = self.data_set_films[
                        (self.data_set_films[:, 1] == item) & (self.data_set_films[:, 0] == user)]

                    testing_real_rates.append(int(rat_A[:, 2]))

                listed_arr = sorted(distance, key=distance.__getitem__)[:k]

                pred = []
                predic_val = np.mean(all_rates)
                arrb = 0
                abj = 0

                for pr in range(len(testing_items)):
                    for i in listed_arr:
                        vec = comps[i]
                        arrb += (distance[i] * vec[pr])
                        abj += distance[i]

                    predic_val += (arrb / abj)

                    pred.append(predic_val)

                # Now, we can make a rank position for the items both actual and predicted.
                rank_pred = {}
                rank_actual = {}
                index = 0
                for a, b, c in zip(pred, testing_real_rates, testing_items):
                    rank_pred[c] = a
                    rank_actual[c] = b
                    index += 1

                # The adjusted rank
                # List of items sorted
                rank_pred = sorted(rank_pred, key=rank_pred.__getitem__)
                rank_actual = sorted(rank_actual, key=rank_actual.__getitem__)

                # Calculate the Spearman rank correlation
                corr = 0
                for itx in testing_items:
                    ind_a = rank_pred.index(itx)
                    ind_b = rank_actual.index(itx)

                    corr += np.square(ind_a - ind_b)

                len_its = len(testing_items)

                if len_its == 1:
                    corr = 1 - corr
                else:
                    corr = 1 - ((6 * corr) / (len_its * (np.square(len_its) - 1)))

                self.precission_recall(pred, testing_real_rates)
                self.bias_under_over(pred, testing_real_rates)
                print(pred)
                print(testing_real_rates)
                acc_abs = np.mean(np.abs(np.subtract(pred, testing_real_rates)))
                acc_sqr = np.mean(np.square(np.subtract(pred, testing_real_rates)))

                print()
                return [acc_abs, acc_sqr, corr]

            else:
                return np.nan

    def bias_under_over(self, predictions, actual_values):

        for val in range(len(predictions)):

            if predictions[val] != np.NaN:

                if predictions[val] < actual_values[val]:
                    self.bias = (self.bias[0] + 1, self.bias[1])
                elif predictions[val] > actual_values[val]:
                    self.bias = (self.bias[0], self.bias[1] + 1)

    def precission_recall(self, predictions, actual_values):

        predictions = list(map(lambda t: 1 if t >= 4 else 0, predictions))
        actual_values = list(map(lambda t: 1 if t >= 4 else 0, actual_values))

        for val in range(len(predictions)):

            if predictions[val] == 1:

                if actual_values[val] == 1:
                    self.conf_matrix[0][0] += 1
                else:
                    self.conf_matrix[1][0] += 1
            else:
                if actual_values[val] == 1:
                    self.conf_matrix[0][1] += 1
                else:
                    self.conf_matrix[1][1] += 1

    def comp_algorithm(self, j):

        # Get list of all things
        list_things = np.unique(self.data_set_films[:, j])

        # Select a target random user
        thing_id = random.choice(list_things)

        # Get all rates made by the target user
        if self.target_id is not None:
            rates = (self.data_set_films[:, j] == thing_id & self.data_set_films[:, abs(j - 1)] == self.target_id)
        else:
            rates = self.data_set_films[:, j] == thing_id

        rates_thing = self.data_set_films[rates]
        set_other_rated = set(rates_thing[:, abs(j - 1)])

        # Series of users
        things_similars = pd.Series(list_things)

        # Drop the user target from the users list
        things_similars.drop(things_similars.index[thing_id])

        switch_tech = {

            0: ('Cosine simmilarity', self.apply_cosine),
            1: ('Euclidean simmilarity', self.apply_euclidean),
            2: ('Manhattan simmilarity', self.apply_manhattan),
            3: ('Adjusted cosine simmilarity', self.apply_adj_cosine),
            4: ('Pearson simmilarity', self.apply_pearson),

        }

        # View different techniques
        for applier in range(5):
            func = switch_tech.get(applier)

            generations = things_similars.apply(lambda user: func[1](j, user, thing_id, set_other_rated))

            # To sort the indexes according to the values contained (distances between target and each entry)
            sorted_distances = np.argsort(generations)

            # Get the n first entries and count them
            counts = [things_similars[i] for i in sorted_distances[:self.k]]

            print("Similar %s to %s %d using %s: " % ((j == 0 and 'users' or 'films'), (j == 0 and 'user' or 'film'),
                                                      thing_id, func[0]))
            print(counts)
            print("\n")

    def user_based(self, v, k, target_id):
        self.v = v
        self.k = k
        self.target_id = target_id

        print('User-based technique')
        self.comp_algorithm(0)

    def item_based(self, v, k, target_id):
        self.v = v
        self.k = k
        self.target_id = target_id

        print('Item-based technique')
        self.comp_algorithm(1)

    def compute_calculus(self, j, thing_id_a, thing_id_b, set_films_thing):
        # Get things of the neighbour
        rates_neig = self.data_set_films[:, j] == thing_id_a
        rates_user_neig = set(self.data_set_films[rates_neig][:, abs(j - 1)])

        # Get the full outer join films
        set_items = set_films_thing & rates_user_neig

        # List of films
        list_others = list(set_items)

        rates_a = []
        rates_b = []
        for item in list_others:

            rat_A = self.data_set_films[(self.data_set_films[:, abs(j - 1)] == item) & (self.data_set_films[:, j] == thing_id_a)]
            rat_B = self.data_set_films[(self.data_set_films[:, abs(j - 1)] == item) & (self.data_set_films[:, j] == thing_id_b)]
            rates_a.append(int(rat_A[:, 2]))
            rates_b.append(int(rat_B[:, 2]))

        return np.array(rates_a).reshape(1, -1), np.array(rates_b).reshape(1, -1), len(set_items)

    def apply_euclidean(self, j, user_id_a, user_id_b, set_films_user):

        rates_a, rates_b, n_mov = self.compute_calculus(j, user_id_a, user_id_b, set_films_user)

        if (self.v is not None and n_mov < self.v) or n_mov == 0:
            return float('inf')
        else:
            return np.sqrt(np.sum(np.square(np.subtract(rates_a, rates_b))))

    def apply_manhattan(self, j, user_id_a, user_id_b, set_films_user):

        rates_a, rates_b, n_mov = self.compute_calculus(j, user_id_a, user_id_b, set_films_user)

        if (self.v is not None and n_mov < self.v) or n_mov == 0:
            return float('inf')
        else:
            return np.sum(np.abs(np.subtract(rates_a, rates_b)))

    def apply_pearson(self, j, user_id_A, user_id_B, set_films_user):
        rates_A, rates_B, n_mov = self.compute_calculus(j, user_id_A, user_id_B, set_films_user)

        if (self.v is not None and n_mov < self.v) or n_mov == 0:
            return float('inf')
        else:
            avg_a = np.mean(rates_A)
            avg_b = np.mean(rates_B)
            return np.sum((rates_A - avg_a) * (rates_B - avg_b)) / (
                    np.sqrt(np.sum(np.square(rates_A - avg_a))) * np.sqrt(np.sum(np.square(rates_B - avg_b))))

    def apply_cosine(self, j, user_id_A, user_id_B, set_films_user):

        rates_A, rates_B, n_mov = self.compute_calculus(j, user_id_A, user_id_B, set_films_user)

        if (self.v is not None and n_mov < self.v) or n_mov == 0:
            return float('inf')
        else:
            return sc2.cosine_similarity(rates_A, rates_B)

    def apply_adj_cosine(self, j, user_id_a, user_id_b, set_films_user):
        rates_a, rates_b, n_mov = self.compute_calculus(j, user_id_a, user_id_b, set_films_user)

        if (self.v is not None and n_mov < self.v) or n_mov == 0:
            return float('inf')
        else:
            rates_a = rates_a - np.nanmean(rates_a)
            rates_b = rates_b - np.nanmean(rates_b)
            return sc2.cosine_similarity(rates_a, rates_b)

rsj = RecCFSys('ratings.txt')

# For part 1: threshold, z-scores/mean-cent, knn, other_id
'''
for v in range(2, 10):
    for k in range(2, 20):
        print('With user based using K={0} and threshold={1}'.format(k, v))
        rsj.user_based(v, k, None)
print('\n')
for v in range(2, 10):
    for k in range(2, 20):
        print('With item based using K={0} and threshold={1}'.format(k, v))
        rsj.item_based(v, k, None)
print("\n")
'''
# For part 2: % of test, Mean-cent/z-scores, Adj-cosine/pearson, knn
for perc in range(5, 20, 5):
    for norm in [False, True]:
        for sim in ['ddsd', 'adj-cosine']:
            for k in range(2, 20):
                rsj.make_prediction(perc, norm, sim, k)
                
