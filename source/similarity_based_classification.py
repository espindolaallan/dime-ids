"""
Script: similarity_based_classification.py

Description:
This script finds the most similar validation examples to each test example by computing distances/similarities 
(using cosine, Euclidean, or Jaccard similarity) and applying a set of pre-trained models from a Pareto front. 
The results can be used later to analyze and select the best classifier or ensemble for final predictions.

Usage:
Run the script with a specified model pool ID, similarity metric, and number of top similar 
examples to consider.

"""

from EspPipeML import esp_utilities
import os
import joblib
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import time
import numpy as np
import argparse
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import MinMaxScaler

def pareto_constraint_filter(res, threshold=None):
    """
    Filters solutions based on a threshold for each objective.
    The objectives are [Performance, Diversity] in F[:, 0] and F[:, 1].
    
    Parameters
    ----------
    res : pymoo result
        The result object from pymoo's minimize().
    threshold : dict, optional
        Example: {'performance': 0.3, 'diversity': 0.9}
        Solutions whose objective values exceed these thresholds are excluded.

    Returns
    -------
    list_of_solution : list
        List of filenames corresponding to the selected Pareto solutions.
    """

    # Analyze results via our class
    analysis = esp_utilities.ParetoAnalysis(res, deduplicate=True)

    # Gather global pareto front solutions
    global_pf_X, global_pf_F, global_pf_G, global_pf_I = analysis.get_global_pf()

    # Store original values (avoid modifying in the loop)
    original_pf_X, original_pf_F, original_pf_G, original_pf_I = (
        global_pf_X.copy(), global_pf_F.copy(), global_pf_G.copy(), global_pf_I.copy()
    )

    # Initialize variables
    list_of_solution = []
    increment = 0.0  # Start from 0 and increment dynamically

    # Avoid infinite loops by limiting the maximum threshold
    max_perf_thr = 1.0  # Upper bound for performance threshold

    while len(list_of_solution) < 10 and (threshold["performance"] + increment <= max_perf_thr):
        # print(f"Increment: {increment}")

        # Apply threshold filter if provided
        if threshold:
            perf_thr = threshold["performance"] + increment
            div_thr  = threshold["diversity"]

            # print(f"Trying perf_thr={perf_thr}, div_thr={div_thr}")

            # Apply filtering on a copy (avoid modifying original arrays)
            mask_all = (original_pf_F[:, 0] <= perf_thr) & (original_pf_F[:, 1] <= div_thr)
            filtered_pf_G, filtered_pf_I = original_pf_G[mask_all], original_pf_I[mask_all]

            # Generate list of solutions
            list_of_solution = [f'gen_{g}_ind_{i}_c{c}.pkl' for g, i in zip(filtered_pf_G, filtered_pf_I) for c in range(1, 5)]

        # Increment threshold (only if the loop runs again)
        increment += 0.01

    # Final check: If no models found, return an empty list
    if not list_of_solution:
        print("Warning: No models found even after increasing performance threshold!")

    return list_of_solution

def load_pareto_front_models(pool_id, constraints=False):
    print(f"Loading models from pool {pool_id}")
    pool_path = f"../results/models/pool/{pool_id}"
    pareto_front_models = {}

    if constraints:
        print("Loading models based on constraints")
        res = esp_utilities.load_from_pickle(f'../results/nsga2/feature_selection/MultiModalOS-IDS_{pool_id}_proposal_auc_res.pkl')
        list_of_solution = pareto_constraint_filter(res, threshold={'performance': 0.1, 'diversity': 1.0})

    
    for filename in os.listdir(pool_path):
        if filename.endswith(".pkl"):
            if constraints and filename not in list_of_solution:
                continue

            print(f"Loading model from: {filename}")
            parts = filename.split('_')
            gen, ind, clf = parts[1], parts[3], parts[4].split('.')[0]
            key = f"{gen}_{ind}"

            if key not in pareto_front_models:
                pareto_front_models[key] = {'c1': None, 'c2': None, 'c3': None, 'c4': None}

            file_path = os.path.join(pool_path, filename)
            # print(f"Loading model from: {file_path}")

            # print(f"Loading model from: {file_path}")
            # Load the model and store it in the dictionary
            # try:
            #     model = joblib.load(file_path)
            #     pareto_front_models[key][clf] = model  # Store model
            # except Exception as e:
            #     print(f"Warning: Could not load {file_path}. Error: {e}")

            model = joblib.load(file_path)


            pareto_front_models[key][clf] = model  # Store model
        
    pareto_front_models_sorted = {key: pareto_front_models[key] for key in sorted(pareto_front_models)}

    return pareto_front_models_sorted

def find_most_similar_cosine(
    query: np.ndarray | pd.Series | pd.DataFrame,
    candidates: pd.DataFrame,
    topn: int = 5
) -> pd.DataFrame:
    """
    Find the `topn` rows in `candidates` most similar to `query` based on cosine similarity.
    """

    # Convert `query` into a 1D NumPy array
    if isinstance(query, pd.DataFrame):
        # If it's a 1-row DataFrame, take that row as a Series
        if len(query) != 1:
            raise ValueError("`query` DataFrame must have exactly 1 row.")
        query_array = query.iloc[0].to_numpy()
    elif isinstance(query, pd.Series):
        query_array = query.to_numpy()
    else:
        query_array = query

    # Reshape `query_array` into a row vector (1, 47) instead of column vector (47, 1)
    query_array = query_array.reshape(1, -1)

    # Convert all candidate rows into a 2D NumPy array
    candidates_array = candidates.to_numpy()

    # Compute norms once
    query_norm = np.linalg.norm(query_array, axis=1, keepdims=True)  # Ensure correct shape
    candidates_norms = np.linalg.norm(candidates_array, axis=1, keepdims=True)

    # Dot product of each row in `candidates_array` with `query_array`
    dot_products = candidates_array @ query_array.T  # Ensure correct shape

    # Avoid division-by-zero; any row that has norm 0 => similarity = 0
    denominator = candidates_norms * query_norm
    with np.errstate(divide='ignore', invalid='ignore'):
        similarities = np.where(denominator == 0, 0, dot_products / denominator)

    # Get indices of the top `topn` similarities (descending order)
    top_indices = np.argsort(-similarities.flatten())[:topn]
    top_sims = similarities.flatten()[top_indices]

    # Build a result DataFrame
    results = candidates.iloc[top_indices].copy()
    results['proximity_score'] = top_sims

    # Sort by similarity descending
    results.sort_values(by='proximity_score', ascending=False, inplace=True)
    return results

def find_most_similar_euclidean(query, candidates, topn=5, normalize=False):    
    if normalize:
        query_norm, candidates_norm = normalize_features_minmax(query, candidates)
        distances = cdist(candidates_norm, query_norm.reshape(1, -1), metric="euclidean").flatten()
    else:
        distances = cdist(candidates, query.reshape(1, -1), metric="euclidean").flatten()
    top_indices = distances.argsort()[:topn]
    
    results = candidates.iloc[top_indices].copy()
    results["proximity_score"] = distances[top_indices]
    return results

def find_most_similar_jaccard(query, candidates, topn=5):
    # Convert query and candidates to binary (0/1)
    if isinstance(query, pd.DataFrame):
        query = (query > 0).astype(int).to_numpy().reshape(1, -1)
    elif isinstance(query, pd.Series):
        query = (query > 0).astype(int).to_numpy().reshape(1, -1)
    else:
        query = (query > 0).astype(int).reshape(1, -1)

    candidates_np = (candidates > 0).astype(int).to_numpy()  # Convert candidates to binary as well

    # Suppress DataConversionWarning (since we already converted to binary)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DataConversionWarning)
        distances = pairwise_distances(candidates_np, query, metric="jaccard").flatten()

    top_indices = distances.argsort()[:topn]

    results = candidates.iloc[top_indices].copy()
    results["proximity_score"] = distances[top_indices]
    return results

def normalize_features_minmax(x_new, X_valid):
    """
    Applies Min-Max scaling to X_valid (DataFrame) and x_new (NumPy array),
    based on the min/max of X_valid's columns. Returns the normalized x_new
    and X_valid as DataFrames.
    """
    # 1) Fit MinMaxScaler on X_valid
    scaler = MinMaxScaler()
    scaler.fit(X_valid.to_numpy())  # learns min & max per column

    # 2) Transform X_valid -> returns a NumPy array
    X_valid_array = scaler.transform(X_valid.to_numpy())

    # 3) Convert the scaled array back to a DataFrame with the same columns & index
    X_valid_norm = pd.DataFrame(X_valid_array, columns=X_valid.columns, index=X_valid.index)

    # 4) Transform x_new (already a NumPy array)
    x_new_norm = scaler.transform(x_new)  # returns a NumPy array of the same shape

    return x_new_norm, X_valid_norm

def classify_most_similar_validation_points(x_new, models, dataset, similarity_metric='cosine', topn=5, normalize=False):
    # Find the most similar examples
    if similarity_metric == 'cosine':
        most_similar = find_most_similar_cosine(x_new, dataset.X_valid, topn=topn)
    elif similarity_metric == 'euclidean':
        most_similar = find_most_similar_euclidean(x_new, dataset.X_valid, topn=topn, normalize=normalize)
    elif similarity_metric == 'jaccard':
        most_similar = find_most_similar_jaccard(x_new, dataset.X_valid, topn=topn)
    else:
        raise ValueError(f"Invalid similarity metric: {similarity_metric}")

    # Convert DataFrame to NumPy arrays for faster processing
    most_similar_np = most_similar.to_numpy()  # Convert entire DataFrame to NumPy array
    most_similar_indexes_np = most_similar.index.to_numpy()  # Convert index to NumPy
    proximity_scores_np = most_similar['proximity_score'].to_numpy()  # Convert proximity scores

    most_similar = most_similar.drop(columns=['proximity_score'])  # Remove from original DataFrame

    results = []

    def classify_model(model_name, current_model, ind):
        """Function to predict labels & probabilities in parallel."""
        features = current_model.get_booster().feature_names
        y_pred = current_model.predict(most_similar_np[:, [most_similar.columns.get_loc(f) for f in features]])
        y_prob = current_model.predict_proba(most_similar_np[:, [most_similar.columns.get_loc(f) for f in features]])

        return [
            {
                'predictions': y_pred[i],
                'prob_class_0': y_prob[i, 0],
                'prob_class_1': y_prob[i, 1],
                'most_similar_indexes': most_similar_indexes_np[i],
                'proximity_scores': proximity_scores_np[i],
                'model': model_name,
                'ind': ind
            }
            for i in range(len(y_pred))
        ]
        

    # Use ThreadPoolExecutor for parallel execution
    with ThreadPoolExecutor(max_workers=96) as executor:  # Adjust `max_workers` based on CPU cores
        futures = []
        for ind, model_dict in models.items():
            for model_name, current_model in model_dict.items():
                futures.append(executor.submit(classify_model, model_name, current_model, ind))

        # Collect results
        for future in futures:
            results.extend(future.result())

    # Convert the list of dictionaries into a DataFrame once (faster than multiple appends)
    df_results = pd.DataFrame(results)
    return df_results

def classify_most_similar_for_entire_test_set(models, dataset, similarity_metric='cosine', topn=5, pool_id=1, generalization=False, normalize=False, constraints=False):
    start_time = time.time()

    if generalization:
        X = dataset.holdout_X
        y = dataset.holdout_y
        # y_multiclass = dataset.holdout_y_multiclass
    else:
        X = dataset.X_test
        y = dataset.y_test
        # y_multiclass = dataset.y_test_multiclass

    total_queries = X.shape[0]
    count = 1
    # query_left = total_queries
    for i in X.index:
        query_start_time = time.time()
        x_new = X.loc[[i]].to_numpy()
        results = classify_most_similar_validation_points(x_new, models, dataset, similarity_metric, topn, normalize=normalize)
        results.loc[:, 'query_index'] = i

        # Save the results to a CSV file
        path = f"../results/selection/{pool_id}/{similarity_metric}"

        if normalize:
            path += "/normalize"
        if generalization:
            path += "/generalization"
        if constraints:
            path += "/constraints"
        if not os.path.exists(path):
            os.makedirs(path)


        results.to_csv(f"{path}/{i}.csv", index=False)
        # print(f'Pool ID {pool_id}, Metric {similarity_metric}, Processed query index {i} in {time.time() - query_start_time} seconds and {query_left} queries left')

        # Percentage completion
        percent_complete = (count / total_queries) * 100

        now = time.time()
        # Estimate time remaining in hours
        estimated_time_remaining = (((now - start_time) * (total_queries - count)) / 60) / 60
        print(f'{pool_id}, {similarity_metric}, {i}, {now - query_start_time}, {percent_complete:.2f}%, {estimated_time_remaining}')
        #1, euclidean, 31725, 39.086615800857544, 0.17%
        count += 1
        # query_left -= 1
    
    # print(f'Pool ID {pool_id}, Metric {similarity_metric}, Finished processing all queries ({dataset.X_test.shape[0]}) in {time.time() - start_time} seconds')
    print(f'{pool_id}, {similarity_metric}, -,{time.time() - start_time}, finished')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Select best models')
    parser.add_argument('--pool_id', type=int, required=True, help='Pool ID')
    parser.add_argument('--similarity_metric', type=str, required=True, help='Similarity metric')
    parser.add_argument('--topn', type=int, default=5, help='Top N most similar examples')
    parser.add_argument('--generalization', action='store_true', help='Generalization mode')
    parser.add_argument('--normalize', action='store_true', help='Normalize features') # Now only for Euclidean
    parser.add_argument('--constraints', action='store_true', help='Select models for the pool based on constraints')
    args = parser.parse_args()

    # print(f"Pool ID: {args.pool_id}, Similarity Metric: {args.similarity_metric}, Top N: {args.topn}, Generalization: {args.generalization}, Normalize: {args.normalize}, Constraints: {args.constraints}")
    # Load the models from the pareto front
    pareto_front_models = load_pareto_front_models(pool_id=args.pool_id, constraints=args.constraints)

    # print(f"Loaded {len(pareto_front_models)} models from the Pareto front")
    # Load chosen combinations from a pickle file
    chosen_combos = esp_utilities.load_from_pickle('../results/crs/crs_chosen_combos.pkl')
    chosen_combo = chosen_combos[args.pool_id-1]

    # print(f"Chosen combo: {chosen_combo}")
    # Initialize the dataset loader
    dataset = esp_utilities.DatasetLoader()

    # print("Loading dataset...")
    # Split the dataset into training and validation sets based on the current combo
    dataset.split_train_validation_ga(holdout_attack_indexes=chosen_combo)
    # Prepare the dataset for retrain the models with the full training set
    dataset.retrain_with_full_data()

    # print("Dataset loaded")
    # Classify the most similar examples for the entire test set
    classify_most_similar_for_entire_test_set(pareto_front_models, 
                                              dataset, similarity_metric=args.similarity_metric, 
                                              topn=args.topn, pool_id=args.pool_id, 
                                              generalization=args.generalization, 
                                              normalize=args.normalize, constraints=args.constraints)
