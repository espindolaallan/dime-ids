# problem_statement.py
# --------------------
# Problem Statement:
# 1. Load the 30 combos
#    1.1 Load the training and evaluation datasets according to the combo
# 2. Train baseline models
#    2.1 Train models using all features - one model for each combo
#    2.2 Train models using OS feature set (windows, linux) - two models for each combo
#    2.3 Train models using feature sets (windows_user_activity, windows_system_activity, linux_network, linux_host) - four models for each combo
# 3. Evaluate the models
#    3.1 Accuracy
#    3.2 Precision
#    3.3 Recall
#    3.4 F1
#    3.5 AUC
#    3.6 Confusion matrix
#    3.7 Attackwise Accuracy
# Save the results for later comparison

# --------------------
# problem_statement_evaluation.py
# --------------------
# 4. Compare the results
#    4.1 Compare the baseline models
#
# --------------------------------------------------

import pandas as pd
from EspPipeML import esp_utilities
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import argparse
import os
import time
#import random forest and mlp from sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def get_classifier(device, scale_pos_weight, classifier):
    """
    Initialize an XGBoost classifier with GPU/CPU settings.
    """
    # Determine tree method and device for XGBoost
    tree_method, xgb_device = esp_utilities.configure_gpu_xgb(device)

    if classifier == 'rf':
        # Random Forest Classifier
        clf = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
    elif classifier == 'mlp':
        # Multi-layer Perceptron Classifier
        clf = MLPClassifier(
            hidden_layer_sizes=(100, 100),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42,
            early_stopping=True
            )
    elif classifier == 'xgb':
        # XGBoost Classifier
        clf = XGBClassifier(
                random_state=42, 
                tree_method=tree_method,
                device=xgb_device,
                scale_pos_weight=scale_pos_weight
            )
    else:
        raise ValueError(f"Unsupported classifier: {classifier}")

    return clf

def train_and_evaluate_model(dataset, device, feature_set, approach, scale_pos_weight, trained_model=None, classifier='xgb'):
    """
    Train and evaluate an XGBClassifier using the specified feature_set and approach.
    Returns:
      - df_metrics: DataFrame containing global metrics
      - df_aw: DataFrame containing attack-wise accuracy
      - clf: Fitted XGBClassifier
    """
    
    if trained_model is None:
        # Train the model
        clf = get_classifier(device, scale_pos_weight, classifier=classifier)
        # Fit the model
        clf.fit(dataset.X_train[feature_set], dataset.y_train.values.ravel())
        # Load the test data
        evaluate_X = dataset.X_test[feature_set]
        evaluate_y = dataset.y_test
        evaluate_y_multiclass = dataset.y_test_multiclass
    else:
        # Load the trained model
        clf = trained_model
        # Load the holdout data
        evaluate_X = dataset.holdout_X[feature_set]
        evaluate_y = dataset.holdout_y
        evaluate_y_multiclass = dataset.holdout_y_multiclass

    # Predict the labels and probabilities
    y_pred = clf.predict(evaluate_X)
    y_prob = clf.predict_proba(evaluate_X)[:, 1]

    # Compute the evaluation metrics
    accuracy = accuracy_score(evaluate_y, y_pred)
    auc = roc_auc_score(evaluate_y, y_prob)
    f1 = f1_score(evaluate_y, y_pred)
    precision = precision_score(evaluate_y, y_pred)
    recall = recall_score(evaluate_y, y_pred)
    confusion = confusion_matrix(evaluate_y, y_pred)

    # Compute attack-wise accuracy
    df_aw = esp_utilities.compute_attackwise_accuracy(evaluate_y, y_pred, evaluate_y_multiclass)
    df_aw['approach'] = approach

    # Construct a DataFrame of global metrics
    df_metrics = pd.DataFrame({
        'metric': ['accuracy', 'auc', 'f1', 'precision', 'recall', 'confusion', 'approach'],
        'baseline': [accuracy, auc, f1, precision, recall, confusion, approach]
    })

    return df_metrics, df_aw, clf

def run_problem_statement_evaluation(chosen_combos, dataset, device, output_path, classifier, generalization=False):
    """
    Iterate over the 30 combos, train baseline models (all_features, OS-based,
    and detailed feature sets), then store results for each pool.
    """
    start_time = time.time()
    # enumerate the itaration to get the index and the chosen combo
    for idx, chosen_combo in enumerate(chosen_combos, start=1):
        print(f"Running evaluation for combo {idx} - {chosen_combo}")
        # 1) Split data for this combo
        dataset.split_train_validation_ga(holdout_attack_indexes=chosen_combo)
        # Prepare the dataset for retrain the models with the full training set
        dataset.retrain_with_full_data()

        # 2) Compute scale_pos_weight = (#neg / #pos)
        count_class_0, count_class_1 = dataset.y_train.value_counts()
        scale_pos_weight = count_class_0 / count_class_1

        # 3) Define baselines to train
        baselines = ['all_features', 'os_features', 'feature_sets']
        results_baseline = {}

        # Define the output file name
        if classifier == 'xgb':
            file = f'baseline_results_pool_{idx}.pkl'
        else:
            file = f'baseline_results_pool_{idx}_{classifier}.pkl'
        
        full_path = os.path.join(output_path, file)

        # Load trained models if generalization
        if generalization:
            trained_models = esp_utilities.load_from_pickle(full_path)

        # 4) Train and evaluate each baseline approach
        for baseline in baselines:
            training_time = time.time()
            metrics_dict = {}
            aw_dict = {}
            models_dict = {}

            if baseline == 'all_features':
                # Single pass over all features
                feature_set = dataset.features
                if generalization:
                    df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, trained_model=trained_models[baseline]['clf'], classifier=classifier)
                else:
                    df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, classifier=classifier)


            elif baseline == 'os_features':
                # Separate pass per OS (windows, linux)
                for os_label in dataset.os_feature_sets.keys():
                    feature_set = dataset.os_feature_sets[os_label]
                    if generalization:
                        df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, trained_model=trained_models[baseline]['clf'][os_label], classifier=classifier)
                    else:
                        df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, classifier=classifier)

                    metrics_dict[os_label] = df_metrics
                    aw_dict[os_label] = df_aw
                    models_dict[os_label] = clf
                
                # Combine results from both OS sets
                df_metrics = pd.concat(metrics_dict.values(), ignore_index=True)
                df_aw = pd.concat(aw_dict.values(), ignore_index=True)
                clf = models_dict

            elif baseline == 'feature_sets':
                # Four sets: windows_user_activity, windows_system_activity, linux_network, linux_host
                for view in dataset.feature_sets.keys():
                    feature_set = dataset.feature_sets[view]

                    if generalization:
                        df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, trained_model=trained_models[baseline]['clf'][view], classifier=classifier)
                    else:
                        df_metrics, df_aw, clf = train_and_evaluate_model(dataset, device, feature_set, approach=baseline, scale_pos_weight=scale_pos_weight, classifier=classifier)
                    metrics_dict[view] = df_metrics
                    aw_dict[view] = df_aw
                    models_dict[view] = clf
                
                # Combine results from all four feature sets
                df_metrics = pd.concat(metrics_dict.values(), ignore_index=True)
                df_aw = pd.concat(aw_dict.values(), ignore_index=True)
                clf = models_dict
        
            # 5) Save results under the baseline key
            results_baseline[baseline] = {
                'metrics': df_metrics,
                'aw': df_aw,
                'clf': clf
            }
        # Add generalization reictory at the end of the path, before file name
        if generalization:
            output_path_ = os.path.join(output_path, 'generalization')
            full_path = os.path.join(output_path_, file)

        # 6) Save pickled dictionary containing metrics & models for this combo
        esp_utilities.save_to_pickle(results_baseline, full_path)
        index_time = time.time()
        print(f"Time taken for combo {idx}: {index_time - training_time:.2f} seconds")
        print('-----------------------------------')
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Total time taken: {elapsed_time:.2f} minutes")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='../results/crs/crs_chosen_combos.pkl', help='Path to the combo pickle file.')
    parser.add_argument('--output', type=str, default='../results/baseline', help='Path to the output results pickle file.')
    parser.add_argument('--device', type=str, default='cuda:2', help='Device to use for training the models.')
    parser.add_argument('--generalization', action='store_true', help='Flag to indicate if the evaluation is for generalization.')
    parser.add_argument('--classifier', type=str, default='xgb', help='Classifier to use for training. Options: xgb, rf, mlp.')
    args = parser.parse_args()

    # Load the 30 combos
    chosen_combos = esp_utilities.load_from_pickle(args.input)

    # Load the dataset
    dataset = esp_utilities.DatasetLoader()

    # Run the problem statement evaluation
    run_problem_statement_evaluation(chosen_combos, dataset, device=args.device, output_path=args.output, generalization=args.generalization, classifier=args.classifier)