# -------------------------------------------------------------------------------
# Script:        Feature and Model Parameter Optimization using NSGA-II
# Author:        Allan Espindola
# Created Date:  14/08/2023
# Description:   This script performs joint optimization of feature combinations 
#                and model parameters using the Non-dominated Sorting Genetic 
#                Algorithm II (NSGA-II). It systematically explores various 
#                feature subsets and their corresponding optimal model parameters 
#                to achieve the best trade-off between model performance and 
#                complexity.
# -------------------------------------------------------------------------------


import numpy as np
import random
from pymoo.core.crossover import Crossover
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from multiprocessing import Pool
# from pathos.multiprocessing import ProcessingPool as Pool
from pymoo.core.callback import Callback
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
from EspPipeML import esp_utilities
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from fastai.tabular.all import *
from sklearn.tree import DecisionTreeClassifier
from icecream import ic
import time
import copy


class EnsembleSamplingNoReposition(Sampling):
     def _do(self, problem, n_samples, **kwargs):
        
        # Define the function to set the chromosome for each classifier
        def setting_chromsome_for_each_clf(clf_name, problem):
            # Define the dictionary to map the classifier name to the view
            clf_name_dict = {
                'c1': 'windows_user_activity', 
                'c2': 'windows_system_activity', 
                'c3': 'linux_network', 
                'c4': 'linux_host'
                }
            # Get the view for the specified classifier
            try:
                view = clf_name_dict[clf_name]
            except KeyError:
                raise ValueError("Unknown classifier name")

            # Get the maximum number of features for the view
            max_features = problem.feature_quantities[view]
            range_features = range(*problem.feature_ranges[view]) # range of features for the view
            X = np.empty((0, max_features), dtype=int) # create empty matrix (population)

            # Generate the number of features for each sample (chromosome)
            # vec_n_features = np.random.randint(low=max(1, problem.f_min), high=max_features, size=n_samples)
            vec_n_features = np.random.randint(
                low=max(1, problem.f_min), 
                high=max_features + 1, 
                size=n_samples
            )

            # Create feature set for each sample (chromosome)
            for n_features in vec_n_features:
                selected_features = np.random.permutation(range_features)[:n_features] # select random features (without repetition)
                chromosome = np.full(max_features, -1, dtype=int) # create chromosome with -1 values
                # chromosome[:n_features] += selected_features # add selected features to the chromosome # this is a bug
                chromosome[:n_features] = selected_features  # Assign selected features

                # Debugging: Check chromosome generation
                # print(f"Classifier {clf_name}, n_features={n_features}, chromosome={chromosome}")

                if len(selected_features) < problem.f_min:
                    raise ValueError(f"Chromosome has fewer than f_min features: {chromosome}")
                X = np.vstack((X, chromosome)) # add chromsome to the matrix (population)
            return X
        
        X_list = []
        # Create the chromosome for each classifier
        for clf_name in ['c1', 'c2', 'c3', 'c4']:
            # Create the chromosome for the specified classifier
            X_list.append(setting_chromsome_for_each_clf(clf_name, problem))
        # Concatenate the chromosomes into a single matrix
        X = np.hstack(X_list)
        return X


class FullIdentityPreservingCrossover(Crossover):
    def __init__(self, prob=0.9, **kwargs):
        super().__init__(n_parents=2, n_offsprings=2, prob=prob, **kwargs)

    def _do(self, problem, X, **kwargs):
        n_parents, n_matings, n_var = X.shape
        Q = np.empty((self.n_offsprings, n_matings, n_var), dtype=int)

        for mating in range(n_matings):
            # Create the offsprings matrix for the current mating
            offsprings = np.empty((self.n_offsprings, n_var), dtype=int)

            # Debug: Start of mating
            # print(f"\n=== Mating {mating + 1}/{n_matings} ===")  # Debug: Start of mating

            # Iterate over each view
            for view in problem.feature_ranges.keys():
                start, end = problem.feature_ranges[view] # Get the start and end positions of the features for the current view
                
                p1, p2 = X[0, mating, start:end], X[1, mating, start:end] # Get the parents for the current view
                # Debugging: Check the view and the parents
                # print(f"\nView: {view}")
                # print(f"Parent 1 (raw): {p1}")
                # print(f"Parent 2 (raw): {p2}")

                p1, p2 = p1[p1 != -1], p2[p2 != -1] # Remove empty features (-1)
                # Debugging: Check the view and the parents after filtering
                # print(f"Parent 1 (filtered): {p1}")
                # print(f"Parent 2 (filtered): {p2}")

                common_elements, diff_elements = self._preprocess_for_crossover(p1, p2)
                # Debugging: Check the common and different elements
                # print(f"Common Elements: {common_elements}")
                # print(f"Different Elements: {diff_elements}")

                
                # Crossover for the current view
                offsprings_part = self._crossover([p1, p2], common_elements, diff_elements)
                # Debugging: Check the offsprings for the current view
                # print(f"Offsprings (raw): {offsprings_part}")

                # Insert the offspring into the correct position in the offsprings matrix
                offsprings[:, start:end] = self._padding_offspring(offsprings_part, problem, view)
                # Debugging: Check the offsprings for the current view (padded)
                # print(f"Offsprings (padded): {offsprings[:, start:end]}")

            # Add the offsprings to the final matrix
            Q[:, mating, :] = offsprings
        return Q

    def _get_diff_and_shuffle(self, parent_features, common_elements):
        #diff elements
        p1 = np.setdiff1d(parent_features[0], common_elements)
        p2 = np.setdiff1d(parent_features[1], common_elements)
        #shuffle
        np.random.shuffle(p1)
        np.random.shuffle(p2)
        return [p1, p2]

    def _preprocess_for_crossover(self, p1, p2):
        parent_features = [p1, p2] # only features
        common_elements = np.intersect1d(parent_features[0], parent_features[1])
        # Get diff elements and shuffle them 
        diff_elements = self._get_diff_and_shuffle(parent_features, common_elements)
        return common_elements, diff_elements

    def _padding_offspring(self, offsprings, problem, view):
        pad_len_1 = problem.feature_quantities[view] - offsprings[0].shape[0]
        pad_len_2 = problem.feature_quantities[view] - offsprings[1].shape[0]

        offspring_1 = np.concatenate((offsprings[0], np.full(pad_len_1, -1)))
        offspring_2 = np.concatenate((offsprings[1], np.full(pad_len_2, -1)))

        return [offspring_1, offspring_2]

    def _crossover(self, parents, common_elements, diff_elements):
        # Calculate lengths of the difference sets for both parents
        len_diff_1, len_diff_2 = len(diff_elements[0]), len(diff_elements[1])
        
        # Determine the crossover point as a random percentage (30%-100%) of the max length
        crossover_point = int(max(len_diff_1, len_diff_2) * np.random.uniform(0.3, 1.0) + 0.5)
        
        # Split the difference elements into parts to cross over
        to_cross_1, to_keep_1 = diff_elements[0][:crossover_point], diff_elements[0][crossover_point:]
        to_cross_2, to_keep_2 = diff_elements[1][:crossover_point], diff_elements[1][crossover_point:]

        # Combine parents' common elements and crossed-over parts
        offspring_1 = np.concatenate((common_elements, to_keep_1, to_cross_2))
        offspring_2 = np.concatenate((common_elements, to_keep_2, to_cross_1))
        
        return [offspring_1, offspring_2]


class AddDeleteReplaceFeatureMutation(Mutation):
    def __init__(self, prob=0.3, prob_var=None, **kwargs) -> None:
        super().__init__(prob, prob_var, **kwargs)
        self.prob = prob

    def _do(self, problem, X, **kwargs):
        X_by_clf = {view: X[:, start:end] for view, (start, end) in problem.feature_ranges.items()}
        X_list = []

        # Iterate through each classifier and its respective feature range
        for clf_name, (view, X_clf) in zip(['c1', 'c2', 'c3', 'c4'], X_by_clf.items()):
            
            # Debugging: Check the view and the chromosome
            # print(f'View: {view}, X_clf: {X_clf}')

            # Delete, Replace, and Add Features
            _X = self._delete(X_clf, problem, view)
            _X = self._replace(_X, problem, view)
            _X = self._add(_X, problem, view)

            # Append the new chromosome to the list
            X_list.append(_X)

        # Concatenate the chromosomes into a single matrix
        return np.concatenate(X_list, axis=1)
    
    
    def prob_decision(self): # Decision based on probability
        return np.random.rand() < self.prob    

    def _add(self, X, problem, view):
        max_size = problem.feature_quantities[view]
        _X = X.copy()
        all_features = set(range(*problem.feature_ranges[view]))

        for k in range(_X.shape[0]):
            features = _X[k] # chromosome
            features = features[features != -1] #only valid features
            len_features = features.shape[0]
            available_features = list(all_features - set(features))
            
            if len_features < max_size and self.prob_decision(): #conditions: size and prob true
                selected_feature = np.random.choice(available_features) # select random feature from available features
                features = np.append(features, selected_feature) # add selected feature

            chromosome = np.array(features, dtype=int)  # Convert to NumPy array
            _X[k] = self._pedding_chromosome(chromosome, problem, view)
        return _X
    
    # Delete features from the chromosome with probability prob
    def _delete(self, X, problem, view):
        min_size = max(1, problem.f_min) # Ensure at least 1 feature remains
        _X = X.copy()

        for k in range(_X.shape[0]):
            features = _X[k] # chromosome
            features = features[features != -1] # only valid features
            len_features = features.shape[0] # number of features
            _features = []

            # Delete features with probability prob
            for i in features:
                if len_features > min_size and self.prob_decision(): # conditions: size and prob true
                    len_features -= 1
                else:
                    _features.append(i)

            chromosome = np.array(_features, dtype=int)
            _X[k] = self._pedding_chromosome(chromosome, problem, view)
        return _X

    # Replace features from the chromosome with probability prob   
    def _replace(self, X, problem, view):
        _X = X.copy()
        # Get all features for the view
        all_features = set(range(*problem.feature_ranges[view]))

        for k in range(_X.shape[0]):
            features = _X[k] # chromosome
            features = features[features != -1]  # only valid features
            _features = []
            available_features = list(all_features - set(features))

            for i in features:
                if self.prob_decision() and available_features:  # conditions: prob and available features
                    selected_feature = random.choice(available_features) # select random feature from available features
                    available_features.remove(selected_feature)  # update available list of features
                    _features.append(selected_feature) # add selected feature
                else: # keep the feature
                    _features.append(i)

            chromosome = np.array(_features, dtype=int)  # Convert to NumPy array
            _X[k] = self._pedding_chromosome(chromosome, problem, view)
        return _X

    def _pedding_chromosome(self, chromosome, problem, view):
        pad_len = problem.feature_quantities[view] - chromosome.shape[0]
        _chromosome = np.concatenate((chromosome, np.full(pad_len, -1)))
        return _chromosome


class EnsembleSelectionProblem(Problem):
    def __init__(self, X_train, y_train, X_test, y_test, feature_set_quantities, feature_set_ranges,f_min,
                 feature_names, cores=1, fitness_metric='auc', device='cuda', **kwargs):           
        super().__init__(
            n_var=sum(feature_set_quantities.values()), 
            n_obj=2, 
            n_constr=0,
            )
        
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.f_min = f_min
        self.feature_quantities = feature_set_quantities
        self.feature_ranges = feature_set_ranges
        self.feature_names = feature_names
        self.cores = cores
        self.fitness_metric = fitness_metric
        self.device = device
        self.cache = {}

    def _evaluate(self, X, out, *args, **kwargs):
        with Pool(processes=self.cores) as pool:
            results = pool.starmap(evaluate_a_chromosome, [(x, self.X_train, self.y_train, self.X_test, self.y_test, 
                                                            self.feature_ranges, self.feature_names, self.fitness_metric, 
                                                            self.device) for x in X])
        
        performance_score, disagreement_score = zip(*results)
        out["F"] = np.column_stack([performance_score, disagreement_score])


def evaluate_a_chromosome(x, X_train, y_train, X_test, y_test, 
                          feature_ranges, feature_names, fitness_metric, device='cuda', return_models=False):
    # Configure GPU for XGBoost
    tree_method, xgb_device = esp_utilities.configure_gpu_xgb(device)
    
    def train_and_evaluate_sklearn_models(
    clf: XGBClassifier, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        clf.fit(X_train, y_train.values.ravel())
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        return y_prob, y_pred, clf

    # Split the chromosome by classifier
    x_by_classifier = {}
    for key, (start, end) in feature_ranges.items():
        x_by_classifier[key] = x[start:end]
    x_decoded = {}
    X_train_selected = {}
    X_test_selected = {}

    # Get the class weights for the XGBoost classifier
    if isinstance(y_train, pd.Series):
        count_class_0, count_class_1 = y_train.value_counts()
    else:
        count_class_0 = np.sum(y_train == 0)
        count_class_1 = np.sum(y_train == 1)
    scale_pos_weight = count_class_0 / count_class_1

    # Initialize dictionaries to store probabilities and predictions
    y_prob_xgb, y_pred_xgb = {}, {}

    models = {}

    # Iterate over feature_ranges and classifier names
    for view, clf_name in zip(feature_ranges.keys(), ['c1', 'c2', 'c3', 'c4']):
        xi = x_by_classifier[view]
        x_decoded[clf_name] = XDecode(x=xi, clf_name=clf_name, feature_names=feature_names, xzao=x_by_classifier)
        X_train_selected[clf_name] = X_train.iloc[:, x_decoded[clf_name].features]
        X_test_selected[clf_name] = X_test.iloc[:, x_decoded[clf_name].features]
        
        if len(x_decoded[clf_name].features) == 0: # if not np.any(x_decoded[clf_name].features):
            raise ValueError(f"No features selected for classifier {clf_name}, x_decoded: {x_decoded[clf_name].features}")
        
        
        clf_xgb = XGBClassifier(
            random_state=42, 
            tree_method=tree_method,
            device=xgb_device,
            scale_pos_weight=scale_pos_weight
        )
        y_prob_xgb[clf_name], y_pred_xgb[clf_name], models[clf_name] = train_and_evaluate_sklearn_models(
            clf_xgb, X_train_selected[clf_name], y_train, X_test_selected[clf_name]
        )
        # models[clf_name] = copy.deepcopy(clf_xgb)

    mean_probs = np.mean(list(y_prob_xgb.values()), axis=0)
    # all_preds = np.array(list(y_pred_xgb.values()))
    # y_vote = np.sum(all_preds, axis=0) > (all_preds.shape[0] / 2)

    if fitness_metric == 'auc':
        performance_score = 1 - roc_auc_score(y_test, mean_probs)
    # elif fitness_metric == 'f1':
    #     performance_score = 1 - f1_score(y_test, y_vote, average='weighted')
    else:
        raise ValueError("Unsupported fitness metric")

    predictions = {clf_name: y_pred_xgb[clf_name] for clf_name in ['c1', 'c2', 'c3', 'c4']}
    mean_disagreement = 1 - esp_utilities.ensemble_disagreement(predictions)

    if return_models:
        return models
    else:
        return performance_score, mean_disagreement


class XDecode():
    def __init__(self, x, clf_name, feature_names, eval=False, xzao = None):
        self.features = self.get_features(x, initial_position=0, eval=eval)
        if len(self.features) == 0:
            raise ValueError(f"No features selected for classifier {clf_name} in chromosome {x} xzao: {xzao}")
        self.selected_feature_names = self.get_selected_features(feature_names)

    def get_features(self, x, initial_position=0, eval=False):
        features = x[initial_position:]
        if not eval:
            return features[features != -1]
        else:
            return [int(num) for num in features if num >= 0][:-2]
    
    def get_selected_features(self, feature_names):
        selected_feature_names = [feature_names[i] for i in self.features]
        return selected_feature_names


class MyCallback(Callback):

    def __init__(self, config_name):
        super().__init__()
        self.config_name = config_name
        self.history = []

    def __call__(self, algorithm):
        self.save_history(algorithm)

        self.save_checkpoint(algorithm)

    def save_history(self, algorithm):
        # Add actual population to history
        self.history.append(algorithm.pop)

        # Save history to file
        history_file = "{}_history.pkl".format(self.config_name)
        with open(history_file, 'wb') as f:
            pickle.dump(self.history, f)

    def save_checkpoint(self, algorithm):
        current_gen = algorithm.n_gen
        create_checkpoint = not (current_gen + 1) % 10  #checkpoint every 10th gen
        if create_checkpoint:
            checkpoint_file = "{}_checkpoint_gen_{}.pkl".format(self.config_name, current_gen)
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'algorithm': algorithm, 'current_gen': current_gen}, f)
            self.last_checkpoint_gen = current_gen
            print(f"Checkpoint saved for generation {current_gen}")

def stratified_splitter(df, stratify_col, valid_pct=0.2):
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=valid_pct, random_state=42)
    for train_index, valid_index in stratified_split.split(df, stratify_col):
        return list(train_index), list(valid_index)
