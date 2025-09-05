"""
dynamic_selection_evaluation.py
Aggregates classification outputs from dynamic selection, applies various voting methods (majority, any, soft), and calculates 
evaluation metrics (accuracy, AUC, F1, precision, recall, confusion matrix). It first merges individual model predictions and 
ground truth labels, then groups results by query to compute final metrics for each voting approach. The resulting DataFrame is 
saved to a user-specified CSV path.
"""

import pandas as pd
from EspPipeML import esp_utilities
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import argparse
import os

def evaluation_x_new(df):
    # Compute accuracy
    accuracy_majority = accuracy_score(df['final_prediction_majority'], df['y_true'])
    accuracy_any = accuracy_score(df['final_prediction_any'], df['y_true'])
    accuracy_soft = accuracy_score(df['final_prediction_soft'], df['y_true'])

    # Compute AUC
    auc_majority = roc_auc_score(df['y_true'], df['avg_prob_class_1'])
    auc_any = roc_auc_score(df['y_true'], df['avg_prob_class_1'])
    auc_soft = roc_auc_score(df['y_true'], df['avg_prob_class_1'])
    
    # Compute F1
    f1_majority = f1_score(df['y_true'], df['final_prediction_majority'])
    f1_any = f1_score(df['y_true'], df['final_prediction_any'])
    f1_soft = f1_score(df['y_true'], df['final_prediction_soft'])

    # Compute precision
    precision_majority = precision_score(df['y_true'], df['final_prediction_majority'])
    precision_any = precision_score(df['y_true'], df['final_prediction_any'])
    precision_soft = precision_score(df['y_true'], df['final_prediction_soft'])

    # Compute recall
    recall_majority = recall_score(df['y_true'], df['final_prediction_majority'])
    recall_any = recall_score(df['y_true'], df['final_prediction_any'])
    recall_soft = recall_score(df['y_true'], df['final_prediction_soft'])

    # Compute confusion matrix
    confusion_majority = confusion_matrix(df['y_true'], df['final_prediction_majority'])
    confusion_any = confusion_matrix(df['y_true'], df['final_prediction_any'])
    confusion_soft = confusion_matrix(df['y_true'], df['final_prediction_soft'])

    df_metrics = pd.DataFrame({
            'metric': [
                'accuracy', 'auc', 'f1', 'precision', 'recall', 'confusion'
            ],
            'majority': [
                accuracy_majority, auc_majority, f1_majority,
                precision_majority, recall_majority, confusion_majority
            ],
            'any': [
                accuracy_any, auc_any, f1_any,
                precision_any, recall_any, confusion_any
            ],
            'soft': [
                accuracy_soft, auc_soft, f1_soft,
                precision_soft, recall_soft, confusion_soft
            ]
        })

    # Compute per-attack accuracy (DataFrames for each approach)
    df_majority_aw = esp_utilities.compute_attackwise_accuracy(df['y_true'], df['final_prediction_majority'], df['y_multiclass'])
    df_any_aw = esp_utilities.compute_attackwise_accuracy(df['y_true'], df['final_prediction_any'], df['y_multiclass'])
    df_soft_aw = esp_utilities.compute_attackwise_accuracy(df['y_true'], df['final_prediction_soft'], df['y_multiclass'])

    # Label each per-attack table with the approach
    df_majority_aw['approach'] = 'majority'
    df_any_aw['approach'] = 'any'
    df_soft_aw['approach'] = 'soft'

    # Combine them into a single DataFrame
    # (This makes it easy to store in one file with approach as an identifier)
    df_attackwise = pd.concat([df_majority_aw, df_any_aw, df_soft_aw], ignore_index=True)

    return df_metrics, df_attackwise

def aggregate_individual_votes(df_classification):
    """Aggregate individual/chromosome votes using majority, any, and soft voting."""
    # Apply majority voting (hard voting)
    df_majority_vote_ind = df_classification.groupby(['query_index', 'ind']).apply(lambda grp: esp_utilities.majority_voting(grp, 'y_pred')).reset_index()
    df_majority_vote_ind.columns = ['query_index', 'ind', 'final_prediction_majority']

    # Apply "Any" Voting (If any classifier predicts attack, classify as attack)
    df_any_vote_ind = df_classification.groupby(['query_index', 'ind']).apply(lambda grp: esp_utilities.any_voting(grp, 'y_pred')).reset_index()
    df_any_vote_ind.rename(columns={0: 'final_prediction_any'}, inplace=True)

    # Apply probability-based aggregation (soft voting)
    df_soft_vote_ind = df_classification.groupby(['query_index', 'ind']).apply(lambda grp: esp_utilities.soft_voting(grp, 'y_prob_0', 'y_prob_1')).reset_index()

    # Merge majority voting results
    df_final_ind = df_majority_vote_ind.merge(df_any_vote_ind, on=['query_index', 'ind'])

    # Merge soft voting results
    df_final_ind = df_final_ind.merge(df_soft_vote_ind, on=['query_index', 'ind'])

    # Merge the ground truth labels
    df_final_ind = df_final_ind.merge(df_classification[['query_index', 'ind', 'y_true']].drop_duplicates(), on=['query_index', 'ind'])

    return df_final_ind

def aggregate_query_votes(df_classification):
    """Aggregate individual votes for each query index (new x)."""
    # Apply majority voting (hard voting)
    df_majority_vote_qr = df_classification.groupby(['query_index']).apply(lambda grp: esp_utilities.majority_voting(grp, 'final_prediction_majority')).reset_index()
    df_majority_vote_qr.columns = ['query_index', 'final_prediction_majority']

    # Apply "Any" Voting (If any classifier predicts attack, classify as attack)
    df_any_vote_qr = df_classification.groupby(['query_index']).apply(lambda grp: esp_utilities.any_voting(grp, 'final_prediction_any')).reset_index()
    df_any_vote_qr.rename(columns={0: 'final_prediction_any'}, inplace=True)

    # Apply probability-based aggregation (soft voting)
    df_soft_vote_qr = df_classification.groupby(['query_index']).apply(lambda grp: esp_utilities.soft_voting(grp, 'avg_prob_class_0', 'avg_prob_class_1')).reset_index()

    # Merge majority voting results
    df_final_qr = df_majority_vote_qr.merge(df_any_vote_qr, on=['query_index'])

    # Merge soft voting results
    df_final_qr = df_final_qr.merge(df_soft_vote_qr, on=['query_index'])

    # Merge the ground truth labels
    df_final_qr = df_final_qr.merge(df_classification[['query_index', 'y_true']].drop_duplicates(), on=['query_index',])

    # Compute accuracies for each approach
    # df_accuracies_ind = df_final_ind.groupby('query_index').apply(compute_accuracies).reset_index()
    # print(df_accuracies_ind)
    return df_final_qr

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Path to the classification results CSV file.')
    parser.add_argument('--output', type=str, help='Path to the output CSV file.')
    parser.add_argument('--generalization', action='store_true', help='Use generalization set instead of test set.')
    args = parser.parse_args()

    # Extract the pool number from the input filename (e.g. classification_results_1_euclidean_accuracy_soft.csv)
    pool = int(args.input.split('_')[2])
    # Load chosen combinations from a pickle file
    chosen_combos = esp_utilities.load_from_pickle('../results/crs/crs_chosen_combos.pkl')
    chosen_combo = chosen_combos[pool-1]

    # Load the dataset
    dataset = esp_utilities.DatasetLoader()

    # Split the dataset into training and validation sets based on the current combo
    dataset.split_train_validation_ga(holdout_attack_indexes=chosen_combo)
    # Prepare the dataset for retrain the models with the full training set
    dataset.retrain_with_full_data()

    # path = f'../results/classification/{args.input}'
    # Load the classification results
    df_classification = pd.read_csv(args.input)
    df_final_ind = aggregate_individual_votes(df_classification)
    df_final_qr = aggregate_query_votes(df_final_ind)

    # Mapping multiclass label for AttackWise Accuracy
    # 1) Convert the y_test_multiclass Series to a dictionary
    #    key: the index, value: the class label
    if args.generalization:
        mapping_dict = dataset.holdout_y_multiclass.to_dict()
    else:
        mapping_dict = dataset.y_test_multiclass.to_dict()
    # 2) Map each row’s query_index to its label
    df_final_qr['y_multiclass'] = df_final_qr['query_index'].map(mapping_dict)
    
    # Evaluation
    df_metrics, df_attackwise = evaluation_x_new(df_final_qr)

    # Prepare the output paths
    dir_part, filename = os.path.split(args.output)
    overall_path = os.path.join(dir_part, f"overall_metrics_{filename}")
    attackwise_path = os.path.join(dir_part, f"attackwise_metrics_{filename}")

    # Save the results
    # df_final_qr.to_csv(f'aggregate_query_votes_{args.output}', index=False)
    df_metrics.to_csv(overall_path, index=False)
    df_attackwise.to_csv(attackwise_path, index=False)



