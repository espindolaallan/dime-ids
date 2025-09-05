from EspPipeML import esp_utilities
import pandas as pd
import os
import time
import joblib
from concurrent.futures import ThreadPoolExecutor
from icecream import ic
import argparse

def load_pareto_front_models(pool_id):
    pool_path = f"../results/models/pool/{pool_id}"
    pareto_front_models = {}

    for filename in os.listdir(pool_path):
        if filename.endswith(".pkl"):
            parts = filename.split('_')
            gen, ind, clf = parts[1], parts[3], parts[4].split('.')[0]
            key = f"{gen}_{ind}"

            if key not in pareto_front_models:
                pareto_front_models[key] = {'c1': None, 'c2': None, 'c3': None, 'c4': None}

            file_path = os.path.join(pool_path, filename)
            # print(f"Loading model from: {file_path}")

            model = joblib.load(file_path)


            pareto_front_models[key][clf] = model  # Store model
    
    pareto_front_models_sorted = {key: pareto_front_models[key] for key in sorted(pareto_front_models)}

    return pareto_front_models_sorted

def overall_scores(df_scores):
    df_scores['final_prediction'] = 'majority'
    df_scores.loc[df_scores['accuracy_any'] > df_scores['accuracy_majority'], 'final_prediction'] = 'any'
    df_scores.loc[df_scores['accuracy_soft'] >= df_scores['accuracy_any'], 'final_prediction'] = 'soft'
    
    return df_scores

def dynamic_selection(df_scores, accuracy_score='accuracy_soft', max_ind=10, random_state=42):
    """
    Select solutions whose `accuracy_score` equals the group's maximum.
    If more than max_ind solutions tie for the top, randomly sample max_ind among them.
    """
    df = df_scores.copy()

    # 1. Identify the max accuracy in each query_index group
    max_scores = df.groupby('query_index')[accuracy_score].transform('max')
    
    # 2. Keep only rows that meet this max accuracy
    df_top = df[df[accuracy_score] == max_scores]

    # 3. Group by query_index again and, if needed, sample up to `max_ind` rows
    def sample_top_ten(group):
        if len(group) <= max_ind:
            return group
        else:
            return group.sample(n=max_ind, random_state=random_state)

    # Apply the sampling and drop the extra group-level index
    df_selected = (
        df_top.groupby('query_index', group_keys=False)
        .apply(sample_top_ten)
    )
    print(f'Selected {df_selected.shape[0]} rows out of {df_top.shape[0]}')

    # 4. Return only the relevant columns
    return df_selected[['query_index', 'ind', accuracy_score]]


def classification_results(models, dataset, df_scores, accuracy_score='accuracy_soft', n_jobs=4, sample_size=None, max_ind=10, generalization=False):
    df_selected = dynamic_selection(df_scores, accuracy_score, max_ind=max_ind)

    if sample_size is not None:
        df_selected = df_selected.sample(n=sample_size, random_state=42)

    grouped = df_selected.groupby('query_index')

    # Define a helper function to process each group in parallel
    def process_group(qidx, group, generalization):
        if generalization:
            x_new = dataset.holdout_X.loc[[qidx]]
            y_new = dataset.holdout_y.loc[qidx]
        else:
            x_new = dataset.X_test.loc[[qidx]]  # Single-row DataFrame for this qidx
            y_new = dataset.y_test.loc[qidx]
        local_results = []

        # For each row in the group
        for idx, row in group.iterrows():
            ind = row['ind']
            model_ind = models[ind]

            for clf_name, model in model_ind.items():
                features = model.get_booster().feature_names
                y_pred = model.predict(x_new[features])        # predicted labels
                y_prob = model.predict_proba(x_new[features])  # predicted probabilities

                # Build a result row: [qidx, ind, clf, y_pred, y_prob_0, y_prob_1, y_true]
                local_results.append([
                    qidx,
                    ind,
                    clf_name,
                    y_pred[0],
                    y_prob[0][0],
                    y_prob[0][1],
                    y_new
                ])
                print(f"Processed query index {qidx}, ind {ind}, clf {clf_name} - y_pred: {y_pred}, y_prob: {y_prob}, y_true: {y_new}, idx: {idx}")

        return local_results

    # Use a ThreadPoolExecutor to process groups in parallel
    result_list = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for qidx, group in grouped:
            futures.append(executor.submit(process_group, qidx, group, generalization))

        # Gather all partial results
        for future in futures:
            result_list.extend(future.result())

    # Convert the accumulated list of rows into a DataFrame
    df_results = pd.DataFrame(
        result_list,
        columns=['query_index', 'ind', 'clf', 'y_pred', 'y_prob_0', 'y_prob_1', 'y_true']
    )
    return df_results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Dynamic selection of models and classification of test data')
    # parser.add_argument('--pool_id', type=int, required=True, help='Pool ID')
    parser.add_argument('--pools', type=str, default='1', help='Pools to aggregate results from.')
    parser.add_argument('--similarity_metric', type=str, default='euclidean', help='Similarity metric to use for selection')
    parser.add_argument('--accuracy_score', type=str, default='accuracy_soft', help='Which accuracy column to use? (accuracy_soft, accuracy_any or accuracy_majority)')
    parser.add_argument('--n_jobs', type=int, default=4, help='Number of parallel jobs to run')
    parser.add_argument('--max_ind', type=int, default=10, help='Maximum number of individuals/solition to select to classify a new query/x')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of samples to process')
    parser.add_argument('--generalization', action='store_true', help='Use the generalization dataset.')
    parser.add_argument('--normalize', action='store_true', help='Normalize the data before computing distances.')
    parser.add_argument('--constraints', action='store_true', help='Use selected models based on constraints.')
    args = parser.parse_args()

    # Define the pools to aggregate results from
    if args.pools == 'all':
        pools = list(range(1, 31))
    elif len(args.pools) == 1: # Single pool
        pools = [int(args.pools)]
    elif args.pools[0] == '[' and args.pools[-1] == ']':  # Set of pools
        pools = list(map(int, args.pools[1:-1].split(',')))
    else:
        raise ValueError('Invalid pool. Options: single (1, 2 and so on) all (1-30), or set [1, 4, 6, 10]')

    for pool in pools:
        print(f'Aggregating results from pool {pool} with metric {args.similarity_metric}')

        # Load the models from the pool
        models = load_pareto_front_models(pool)

        # Load chosen combinations from a pickle file
        chosen_combos = esp_utilities.load_from_pickle('../results/crs/crs_chosen_combos.pkl')
        chosen_combo = chosen_combos[pool-1]

        # Load the dataset
        dataset = esp_utilities.DatasetLoader()

        # Split the dataset into training and validation sets based on the current combo
        dataset.split_train_validation_ga(holdout_attack_indexes=chosen_combo)
        # Prepare the dataset for retrain the models with the full training set
        dataset.retrain_with_full_data()

        # Prepare the base path
        base_path = '../results/selection/aggregated'

        # Add complementary paths
        if args.generalization:
            base_path += '/generalization'
        if args.normalize:
            base_path += '/normalize'
        if args.constraints:
            base_path += '/constraints'
        
        # Load the scores
        df_scores = pd.read_csv(f'{base_path}/aggregated_scores_{pool}_{args.similarity_metric}.csv')        

        # Run the classification
        df_results = classification_results(models, dataset, df_scores, accuracy_score=args.accuracy_score, 
                                            n_jobs=args.n_jobs, sample_size=args.sample_size, max_ind=args.max_ind, generalization=args.generalization)

        # Save path
        save_path = '../results/classification'
        if args.generalization:
            save_path += '/generalization'
        if args.normalize:
            save_path += '/normalize'
        if args.constraints:
            save_path += '/constraints'

        #Check if the file exists
        if not os.path.exists(save_path):
                os.makedirs(save_path)

        if args.sample_size is not None:
            save_path = f"{save_path}/sample_{args.sample_size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        # Save the results
        df_results.to_csv(f"{save_path}/classification_results_{pool}_{args.similarity_metric}_{args.accuracy_score}.csv", index=False)


