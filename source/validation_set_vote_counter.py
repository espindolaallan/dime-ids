"""
validation_set_vote_counter.py

This script aggregates classification predictions (similarity_based_classification.py outputs) from multiple models 
(across different pools and distance metrics) on a validation set. It then may applies three voting strategies—majority
voting, "any" voting, and soft (probability) voting to derive a final prediction for each sample. Afterward, it 
computes accuracy scores for each voting approach.
"""

from EspPipeML import esp_utilities
import pandas as pd
import os
import time
import argparse
from concurrent.futures import ThreadPoolExecutor

# def majority_voting(group):
#     """Returns the most common prediction in the group (hard voting)."""
#     return group['predictions'].mode()[0]  # Most frequent value

def compute_val_ind_accuracies(group):
    """Computes the accuracy of each individual classifier in the group."""
    accuracy_majority = (group['final_prediction_majority'] == group['ground_truth']).mean()
    accuracy_any = (group['final_prediction_any'] == group['ground_truth']).mean()
    accuracy_soft = (group['final_prediction_soft'] == group['ground_truth']).mean()
    
    return pd.Series({
        'accuracy_majority': accuracy_majority,
        'accuracy_any': accuracy_any,
        'accuracy_soft': accuracy_soft
    })

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Compute and aggregate results from approximations of the validation set.')
    # pools, options: single (1, 2 and so on) all (1-30), or set [1, 4, 6, 10]
    parse.add_argument('--pools', type=str, default='1', help='Pools to aggregate results from.')
    # metric, options: euclidean, cosine, jacard
    parse.add_argument('--metric', type=str, default='euclidean', help='Metric used to compute distances.')
    parse.add_argument('--n_jobs', type=int, default=1, help='Number of jobs to run in parallel.')
    parse.add_argument('--generalization', action='store_true', help='Use the generalization dataset.')
    parse.add_argument('--normalize', action='store_true', help='Normalize the data before computing distances.')
    parse.add_argument('--constraints', action='store_true', help='Use selected models based on constraints.')

    args = parse.parse_args()

    start_time = time.time()

    # Define the pools to aggregate results from
    if args.pools == 'all':
        pools = list(range(1, 31))
    elif len(args.pools) == 1: # Single pool
        pools = [int(args.pools)]
    elif args.pools[0] == '[' and args.pools[-1] == ']':  # Set of pools
        pools = list(map(int, args.pools[1:-1].split(',')))
    else:
        raise ValueError('Invalid pool. Options: single (1, 2 and so on) all (1-30), or set [1, 4, 6, 10]')

    # Define the metric to aggregate results from
    if args.metric in ['euclidean', 'cosine', 'jaccard']:
        metric = [args.metric]
    elif args.metric == 'all':
        metric = ['euclidean', 'cosine', 'jaccard']
    else:
        raise ValueError('Invalid metric. Options: euclidean, cosine, jaccard, all')

    for pool in pools:
        for m in metric:
            path = f'../results/selection/{pool}/{m}'

            if args.normalize:
                path += '/normalize'
            if args.generalization:
                path += '/generalization'
            if args.constraints:
                path += '/constraints'
            print(f'Aggregating results from pool {pool} with metric {m} \ngeneralization: {args.generalization} \nnormalize: {args.normalize} \nconstraints: {args.constraints}')


            # Get the list of files in the folder
            files = [f for f in os.listdir(path) if f.endswith('.csv')]
            print(f'Oppenning {len(files)} files')
            # Initialize the dataframe
            df = pd.DataFrame()

            with ThreadPoolExecutor(max_workers=args.n_jobs) as executor:
                futures = [
                    executor.submit(pd.read_csv, os.path.join(path, file))
                    for file in files
                ]
                df_list = [future.result() for future in futures]
            df = pd.concat(df_list)

            print(f'Files merged in {(time.time() - start_time) / 60 } minutes')
            partial_time = time.time()

            # Apply majority voting (hard voting)
            df_majority_vote = df.groupby(['query_index', 'ind', 'most_similar_indexes']).apply(esp_utilities.majority_voting).reset_index()
            # df_majority_vote.rename(columns={0: 'final_prediction_majority'}, inplace=True)
            df_majority_vote.columns = ['query_index', 'ind', 'most_similar_indexes', 'final_prediction_majority']

            print('Majority voting done in ', (time.time() - partial_time) / 60, ' minutes')
            partial_time = time.time()

            # Apply "Any" Voting (If any classifier predicts attack, classify as attack)
            df_any_vote = df.groupby(['query_index', 'ind', 'most_similar_indexes']).apply(esp_utilities.any_voting).reset_index()
            df_any_vote.rename(columns={0: 'final_prediction_any'}, inplace=True)

            print(f'Any voting done in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()

            # Apply probability-based aggregation (soft voting)
            df_soft_vote = df.groupby(['query_index', 'ind', 'most_similar_indexes']).apply(esp_utilities.soft_voting).reset_index()

            print(f'Soft voting done in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()

            # Merge majority voting results
            df_final = df_majority_vote.merge(df_any_vote, on=['query_index', 'ind', 'most_similar_indexes'])

            # Merge soft voting results
            df_final = df_final.merge(df_soft_vote, on=['query_index', 'ind', 'most_similar_indexes'])

            print(f'Merging done in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()

            # Load chosen combinations from a pickle file
            chosen_combos = esp_utilities.load_from_pickle('../results/crs/crs_chosen_combos.pkl')
            chosen_combo = chosen_combos[pool-1]

            # Initialize the dataset loader
            dataset = esp_utilities.DatasetLoader()

            # Split the dataset into training and validation sets based on the current combo
            dataset.split_train_validation_ga(holdout_attack_indexes=chosen_combo)
            # Prepare the dataset for retrain the models with the full training set
            dataset.retrain_with_full_data()

            print(f'Dataset loaded in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()

            # Add the ground truth to the DataFrame
            df_final['ground_truth'] = df_final['most_similar_indexes'].map(lambda idx: dataset.y_valid.loc[idx] if idx in dataset.y_valid.index else None)
            print(f'Ground truth added in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()

            # Group by query_index and ind, then apply the function
            df_scores = df_final.groupby(['query_index', 'ind']).apply(compute_val_ind_accuracies).reset_index()
            print(f'Accuracies computed in {(time.time() - partial_time) / 60} minutes')
            partial_time = time.time()
            
            # Prepare the save path
            save_path = f'../results/selection/aggregated'

            # Prepare the save path
            if args.generalization:
                save_path += '/generalization'
            if args.normalize:
                save_path += '/normalize'
            if args.constraints:
                save_path += '/constraints'
            # Create the directory if it does not exist
            os.makedirs(save_path, exist_ok=True)

            # Save the aggregated DataFrames
            df_final.to_csv(f'{save_path}/aggregated_results_{pool}_{m}.csv', index=False)
            df_scores.to_csv(f'{save_path}/aggregated_scores_{pool}_{m}.csv', index=False)

            print(f'Saving done in {(time.time() - partial_time) / 60} minutes')
            print(f'Total time: {(time.time() - start_time) / 60} minutes')