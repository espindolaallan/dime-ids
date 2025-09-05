
"""
NSGA2Proposal.py
Script:        Feature Optimization using NSGA-II
Description:   This script performs optimization of feature combinations 
               using the Non-dominated Sorting Genetic Algorithm II (NSGA-II).  
               It systematically explores various feature subsets to achieve 
               the best trade-off between model performance and diversity.
"""

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from EspPipeML import esp_utilities
import argparse
import warnings
#warnings.filterwarnings("ignore", message="A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.")
# Suprimir apenas o aviso específico de XGBoost sobre dispositivos incompatíveis
warnings.filterwarnings('ignore', category=UserWarning, message=".*Falling back to prediction using DMatrix due to mismatched devices.*")

from my_operations_ensemble import EnsembleSamplingNoReposition, FullIdentityPreservingCrossover, AddDeleteReplaceFeatureMutation, EnsembleSelectionProblem, MyCallback
esp_utilities.fix_seeds(42)

def continue_from_checkpoint(checkpoint_file, remaining_gens, problem):
    algorithm = esp_utilities.load_from_pickle(checkpoint_file)

    # Continue optimization
    res = minimize(
        problem,
        algorithm,
        ('n_gen', remaining_gens),   # This is how many generations are left
        seed=1,
        verbose=True,
        save_history=True,
        callback=MyCallback(checkpoint_file)  # Reuse the callback for further checkpointing
    )

    return res


def run_nsga2(configs, checkpoint_file=None):
    # If a checkpoint is provided, load the previous state and continue
    if checkpoint_file:
        with open(checkpoint_file, 'rb') as f:
            checkpoint_data = pickle.load(f)
            algorithm = checkpoint_data['algorithm']
            last_gen = checkpoint_data['current_gen']
            remaining_gens = configs['n_gen'] - last_gen
            print(f"Resuming from generation {last_gen}. Remaining generations: {remaining_gens}.")
    else:
        # Define the NSGA-II algorithm
        algorithm = NSGA2(
            pop_size=configs['pop_size'],
            n_offsprings=int(100 * configs['rate_offsprings']),
            sampling=EnsembleSamplingNoReposition(),
            mutation=AddDeleteReplaceFeatureMutation(prob=configs['mutation_prob']),
            crossover=FullIdentityPreservingCrossover(),
        )
        remaining_gens = configs['n_gen']

    problem = EnsembleSelectionProblem(
        configs['X_train'],
        configs['y_train'],
        configs['X_test'],
        configs['y_test'],
        configs['feature_set_quantities'],
        configs['feature_set_ranges'],
        configs['f_min'],
        configs['feature_names'],
        configs['cores'],
        configs['fitness_metric'],
        configs['device']
    )

    # Set up the callback with a checkpoint file name
    checkpoint_file_name = configs['check_point_name']
    callback = MyCallback(checkpoint_file_name)
    
    res = minimize(
            problem,
            algorithm,
            ('n_gen', remaining_gens),
            seed=1,
            verbose=True,
            save_history=True,
            callback=callback
        )
    return res

def run_everything(exec_name, checkpoint_file=None, cores=1, fitness_metric='auc', holdout_attacks=None, device='cuda'):
    # Convert the holdout_attacks string to a tuple of integers
    if holdout_attacks is not None:
        holdout_attacks = tuple(map(int, holdout_attacks.split(',')))

    # Load the dataset and split it into training and validation sets
    dataset = esp_utilities.DatasetLoader()
    dataset.split_train_validation_ga(holdout_attack_indexes=holdout_attacks)

    execution_name = '../results/nsga2/feature_selection/'+ exec_name
    check_point_name = '../results/nsga2/checkpoint/feature_selection/'+ exec_name
    
    configs = {
        'execution_name': '',
        'check_point_name': '',
        'y_train': dataset.y_train,
        'X_train': dataset.X_train,
        # 'y_train_multiclass': dataset.y_train_multiclass,
        'y_test': dataset.y_valid,
        'X_test': dataset.X_valid,
        # 'y_test_multi_class': dataset.y_valid_multiclass,
        'n_features': dataset.X_train.shape[1],
        'feature_names': dataset.X_train.columns.tolist(),
        'feature_set_quantities': dataset.get_feature_set_quantities(),
        'feature_set_ranges': dataset.get_feature_set_ranges(),
        'f_min': 1,
        'pop_size': 100, #X.shape[1]*2,
        'n_gen': 100,
        'mutation_prob': 0.1,
        'rate_offsprings': 0.9,
        'cores': cores,
        'fitness_metric': fitness_metric,
        'device': device,
        'execution_name': execution_name + '_proposal_' + fitness_metric,
        'check_point_name': check_point_name + '_proposal_' + fitness_metric
    }


    if checkpoint_file:
        res = run_nsga2(configs, checkpoint_file)
    else:
        res = run_nsga2(configs)

    esp_utilities.save_to_pickle(res, configs['execution_name'] + '_res.pkl')

    print('___________________________________________\n\n')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run machine learning models.')
    parser.add_argument('--exec_name', type=str, default='MultiModalOS-IDS', help='Name of the execution')
    parser.add_argument('--cores', type=int, default=20, help='Number of cores to use')
    parser.add_argument('--metric', type=str, default='auc', help='Name of the fitness metric to be used for optimization')
    parser.add_argument('--device', type=str, default='cuda', help='Device to be used for training')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file to resume optimization')
    parser.add_argument('--holdout_attacks', type=str, default=None, help='List of number of attacks to be used in the holdout set') 
    # parser.add_argument('--holdout_attacks', type=int, nargs='+', default=None, help='List of classes (ints) to be used in the holdout set') # Attacks to be excluded from the training set and testing set to evaluate the generalization of the model
    

    args = parser.parse_args()

    run_everything(exec_name=args.exec_name, cores=args.cores, fitness_metric=args.metric, device=args.device, holdout_attacks=args.holdout_attacks, checkpoint_file=args.checkpoint)

####################
# example of usage #
####################

# 1. Run the following command in the terminal

# start optimization from scratch
# python NSGA2Proposal.py --cores 20 --metric auc --device cuda

# resume optimization from a checkpoint
# python NSGA2Proposal.py --cores 20 --metric auc --device cuda   --checkpoint ../results/nsga2/checkpoint/feature_selection/unsw-nb15_proposal_auc.pkl 

