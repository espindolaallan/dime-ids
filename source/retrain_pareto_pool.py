"""
retrain_pareto_pool.py
----------------------
This script retrains models based on the Pareto front solutions obtained from NSGA-II results.
It iterates over chosen combinations of features, splits the dataset accordingly, and evaluates
each solution in the Pareto front. The trained models are then saved to a specified directory
for use in dynamic selection.
"""

from EspPipeML import esp_utilities
from my_operations_ensemble import evaluate_a_chromosome
import os
import time
from sklearn.utils.validation import check_is_fitted
import joblib    

# Start timing the script execution
start_time = time.time()

# Load chosen combinations from a pickle file
chosen_combos = esp_utilities.load_from_pickle('../results/crs/crs_chosen_combos.pkl')

# Initialize the dataset loader
dataset = esp_utilities.DatasetLoader()

# Iterate over each combination
for combo, idx in zip(chosen_combos, range(len(chosen_combos))):
    # Split the dataset into training and validation sets based on the current combo
    dataset.split_train_validation_ga(holdout_attack_indexes=combo)
    # Prepare the dataset for retrain the models with the full training set
    dataset.retrain_with_full_data()
    idx_combo = idx + 1

    # Load the NSGA-II results for the current combination
    res = esp_utilities.load_from_pickle('../results/nsga2/feature_selection/MultiModalOS-IDS_'+ str(idx_combo) +'_proposal_auc_res.pkl')
    # Perform Pareto analysis on the results
    pareto_analises = esp_utilities.ParetoAnalysis(res)

    # Get the global Pareto front
    g_X, g_F, g_G, g_I = pareto_analises.get_global_pf()

    # Iterate over each solution in the Pareto front
    for loop_count, (x, f, g, i) in enumerate(zip(g_X, g_F, g_G, g_I), start=1):
    # for x, f, g, i in zip(g_X, g_F, g_G, g_I):
        # Convert the solution to integers
        x_int = x.astype(int)

        # Evaluate the chromosome and get the models
        models = evaluate_a_chromosome(
            x=x_int,
            X_train=dataset.X_train, 
            y_train=dataset.y_train, 
            X_test=dataset.X_test, 
            y_test=dataset.y_test, 
            feature_ranges=dataset.get_feature_set_ranges(), 
            feature_names=dataset.features, 
            fitness_metric='auc', 
            device='cuda:2',
            return_models=True
        )
        
        # Define the directory to save the models
        directory_path = f"../results/models/pool/{idx_combo}"

        # Check if the directory exists, if not, create it
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Save the entire dictionary of models in the directory
        # model_path = os.path.join(directory_path, f"gen_{g}_ind_{i}_models.pkl")
        # esp_utilities.save_to_pickle(models, model_path)

        def xgb_is_fitted(model):
            # For XGBClassifier, being "fitted" basically means having a non-empty booster.
            return hasattr(model, "_Booster") and model._Booster is not None

        for clf_name, model in models.items():
            try:
                if xgb_is_fitted(model):
                    model_path = os.path.join(directory_path, f"gen_{g}_ind_{i}_{clf_name}.pkl")
                    joblib.dump(model, model_path)
                else:
                    print(f"❌ Model {clf_name} not fitted! Skipping save.")
            except Exception as e:
                print(f"❌ Model {clf_name} could not be saved! Error: {e}")

        # Print a message indicating the models have been saved
        print(f"Solution number {loop_count} Models gen_{g}_ind_{i} saved in directory: {directory_path}")

# End timing the script execution
end_time = time.time()
elapsed_time = end_time - start_time

# Print elapsed time in seconds
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Print elapsed time in minutes
print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")