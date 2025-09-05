import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import random
import numpy as np
import torch
from itertools import combinations
from typing import Tuple
import numpy as np
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class DatasetLoader:
    windows_user_activity = ['id1', 'id2', 'id5', 'id10', 'id11', 'id12', 'id13', 'id14', 'id15']
    windows_system_activity = ['id3', 'id4', 'id6', 'id7', 'id8', 'id9', 'id16', 'id17', 'id18', 
                               'id19', 'id20', 'id21', 'id22']
    linux_network = ['rxpck', 'txpck', 'rxkB', 'txkB', 'rxcmp', 'txcmp', 'rxmcst', 'ifutil']
    linux_host = ['kbdirty', 'kbkstack', 'kbvmused', 'usr', 'nice', 'sys', 'iowait', 'steal', 
                  'irq', 'soft', 'guest', 'gnice', 'idle', 'pswpin', 'pswpout', 'proc', 'cswch']
    feature_sets = {'windows_user_activity': windows_user_activity, 'windows_system_activity': windows_system_activity, 
                    'linux_network': linux_network, 'linux_host': linux_host}
    os_feature_sets = {'windows': windows_user_activity + windows_system_activity, 'linux': linux_network + linux_host}
    labels = ['classe', 'classe_atk']
    features = windows_user_activity + windows_system_activity + linux_network + linux_host
    cols = features + labels

    attack_columns = ['AcunetixScadaAuthhm', 'AcunetixAutenticado', 'ArachniRodouporh', 'AcunetixDefaultWire', 
                  'AcunetixDefaultXX', 'AcunetixScadaBRDefault', 'SmodDosWriteAllRegisters', 'SmodDosWriteAllcoins', 
                  'SmodGetFunc', 'SmodScannerUID', 'ModFuzzer', 'NexposeExaustiveNoFW', 'NexposeFull', 'NessusDefault', 
                  'Nmap', 'PLCScan']

    def __init__(self, filename='../dataset/MultiModalOS-IDS.csv'):
        self.df_data = pd.read_csv(filename, usecols=self.cols)[self.cols]
        self.df_data = self.df_data.loc[self.df_data['classe_atk'].isin(self.attack_columns + ['normal'])].sort_index()
        self.original_X = self.df_data.drop(self.labels, axis=1)
        self.original_y = self.df_data['classe']
        self.original_y_multiclass = self.df_data['classe_atk']
        self.X = self.original_X.copy()
        self.y = self.original_y.copy()
        self.y_multiclass = self.original_y_multiclass.copy()
        self.X_train = None
        self.y_train = None
        self.y_train_multiclass = None
        self.X_test = None
        self.y_test = None
        self.X_valid = None
        self.y_valid = None
        self.y_valid_multiclass = None
        self.X_new = None
        self.y_new = None
        self.y_new_multiclass = None
        self.holdout_attacks = None
        self.holdout_X = None
        self.holdout_y = None
        self.holdout_y_multiclass = None
        self.X_hidden = None
        self.y_hidden = None
        self.y_hidden_multiclass = None

    def _set_original_data(self):
        self.X = self.original_X.copy()
        self.y = self.original_y.copy()
        self.y_multiclass = self.original_y_multiclass.copy()

    # Set the attacks to be used in the generalization holdout 
    def _set_generalization_holdout_attacks(self, holdout_attack_indexes: Tuple[int]) -> None:
        try:
            self.holdout_attacks = [self.attack_columns[i] for i in holdout_attack_indexes]
        except IndexError:
            raise ValueError("Índices inválidos fornecidos para holdout_attack_indexes")


    def _get_holdout_mask(self):
        """Retorn a boolean mask to identify holdout data."""
        return self.y_multiclass.isin(self.holdout_attacks)

    # Set the data for the generalization holdout
    def _set_generalization_holdout_data(self, holdout_attack_indexes: Tuple[int]) -> None:
        """Remove attacks to holdout and adjusts the amount of normal traffic to maintain the proportion."""
        self._set_generalization_holdout_attacks(holdout_attack_indexes)
        self._set_original_data()

        holdout_mask = self._get_holdout_mask()  # Mask to identify holdout data
        normal_mask = self.y_multiclass == "normal"  # Mask to identify normal traffic

        # Split the data into holdout
        holdout_X_attacks = self.X[holdout_mask]
        holdout_y_attacks = self.y[holdout_mask]
        holdout_y_multiclass_attacks = self.y_multiclass[holdout_mask]

        # Quantity of attacks to be removed
        num_attacks_removed = holdout_mask.sum()

        # Quantity of normal traffic in the dataset
        total_normal = normal_mask.sum()

        # Fraction of attacks removed in relation to the total number of attacks before the holdout
        total_attacks_before_holdout = len(self.y) - total_normal  # Total attacks before the holdout
        attack_ratio = num_attacks_removed / total_attacks_before_holdout

        # Quantity of normal traffic to remove
        normal_to_remove = int(attack_ratio * total_normal)
        normal_to_remove = min(normal_to_remove, total_normal)  # Prevents removing more than available

        # Select a random sample of normal traffic to remove (deterministic)
        normal_indices_to_remove = (
            self.X[normal_mask]
            .sort_index()  # Garante que a ordem dos exemplos seja sempre a mesma
            .sample(n=normal_to_remove, random_state=42)
            .index
        )
        holdout_X_normal = self.X.loc[normal_indices_to_remove]
        holdout_y_normal = self.y.loc[normal_indices_to_remove]
        holdout_y_multiclass_normal = self.y_multiclass.loc[normal_indices_to_remove]

        # Create a new removal mask combining attacks and normal traffic
        combined_removal_mask = holdout_mask | self.X.index.isin(normal_indices_to_remove)

        # Update the holdout set with attacks + normal traffic
        self.holdout_X = pd.concat([holdout_X_attacks, holdout_X_normal]).sort_index()
        self.holdout_y = pd.concat([holdout_y_attacks, holdout_y_normal]).sort_index()
        self.holdout_y_multiclass = pd.concat([holdout_y_multiclass_attacks, holdout_y_multiclass_normal]).sort_index()

        # Remove the data from the main dataset
        self.X = self.X[~combined_removal_mask].sort_index()
        self.y = self.y[~combined_removal_mask].sort_index()
        self.y_multiclass = self.y_multiclass[~combined_removal_mask].sort_index()
        
    def split_train_test(self, test_size=0.3, holdout_attack_indexes=None):
        # If holdout_attack_indexes is not None, set the data for the generalization holdout
        if holdout_attack_indexes is not None:
            self._set_generalization_holdout_data(holdout_attack_indexes)
        else:
            self._set_original_data()
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.y_train_multiclass, self.y_test_multiclass = train_test_split(
            self.X, self.y, self.y_multiclass, test_size=test_size, stratify=self.y_multiclass, random_state=42
        )

    def split_train_test_valid(self, test_size=0.15, valid_size=0.15, holdout_attack_indexes=None):
        if holdout_attack_indexes is not None:
            self._set_generalization_holdout_data(holdout_attack_indexes)
        else:
            self._set_original_data()

        # First split: test set (30% of the total)
        train_proportion = test_size + valid_size
        self.X_train, X_temp, self.y_train, y_temp, self.y_train_multiclass, y_temp_multiclass = train_test_split(
            self.X, self.y, self.y_multiclass, test_size=(train_proportion), stratify=self.y_multiclass, random_state=42
        )

        # Proportion of the validation set
        valid_proportion = valid_size / (test_size + valid_size)
        
        # Second split: validation and test sets
        self.X_valid, self.X_test, self.y_valid, self.y_test, self.y_valid_multiclass, self.y_test_multiclass = train_test_split(
            X_temp, y_temp, y_temp_multiclass, test_size=(1 - valid_proportion), stratify=y_temp_multiclass, random_state=42
        )

    def split_train_validation_ga(self, test_size=0.3, valid_size=0.15, holdout_attack_indexes=None):
        if holdout_attack_indexes is not None:
            self._set_generalization_holdout_data(holdout_attack_indexes)
        else:
            self._set_original_data()

        # First split: test set (30% of the total) - Split the 30% of the data that will be hidden from the GA
        self.X_train, self.X_hidden, self.y_train, self.y_hidden, self.y_train_multiclass, self.y_hidden_multiclass = train_test_split(
            self.X, self.y, self.y_multiclass, test_size=test_size, stratify=self.y_multiclass, random_state=42
        )

        # Proportion of the validation set
        valid_proportion = valid_size / (1 - test_size)

        # Second split: train and validation sets
        self.X_train, self.X_valid, self.y_train, self.y_valid, self.y_train_multiclass, self.y_valid_multiclass  = train_test_split(
            self.X_train, self.y_train, self.y_train_multiclass, test_size=(valid_proportion), stratify=self.y_train_multiclass, random_state=42
        )

    # Get the feature positions for each feature
    def get_feature_positions(self):
        feature_positions = {}
        for set_name, features in self.feature_sets.items():
            feature_positions[set_name] = {feature: self.features.index(feature) for feature in features}
        return feature_positions
    
    # Get the feature set ranges (inclusive) for each feature set
    def get_feature_set_ranges(self):
        feature_set_ranges = {}
        for set_name, features in self.feature_sets.items():
            indices = [self.features.index(feature) for feature in features]
            feature_set_ranges[set_name] = (min(indices), max(indices) + 1)  # +1 to make the range inclusive
        return feature_set_ranges
    
    # Get the feature set ranges (inclusive) for each OS
    def get_feature_set_ranges_os(self):
        feature_set_ranges = {}
        for os, features in self.os_feature_sets.items():
            indices = [self.features.index(feature) for feature in features]
            feature_set_ranges[os] = (min(indices), max(indices) + 1)
        return feature_set_ranges
    
    # Get the quantities of features in each feature set
    def get_feature_set_quantities(self):
        feature_set_quantities = {}
        for set_name, features in self.feature_sets.items():
            feature_set_quantities[set_name] = len(features)
        return feature_set_quantities
    
    # Get the quantities of features in each OS
    def get_feature_set_quantities_os(self):
        feature_set_quantities = {}
        for os, features in self.os_feature_sets.items():
            feature_set_quantities[os] = len(features)
        return feature_set_quantities
    
    def retrain_with_full_data(self):
        """Combine train and validation and use the 30% hidden as the final test set."""

        # Combine train and validation to form the new final training set
        self.X_train = pd.concat([self.X_train, self.X_valid])
        self.y_train = pd.concat([self.y_train, self.y_valid])
        self.y_train_multiclass = pd.concat([self.y_train_multiclass, self.y_valid_multiclass])

        # The 30% hidden are now the new final test
        self.X_test = self.X_hidden
        self.y_test = self.y_hidden
        self.y_test_multiclass = self.y_hidden_multiclass
            
    
def compute_attackwise_accuracy(y_test, y_pred, y_multi_class):
    """
    Compute and return the accuracy of predictions, attack-wise.

    Given the true labels, predicted labels, and corresponding attack categories,
    this function calculates the accuracy of predictions for each unique attack category.
    Additionally, it provides counts of total instances, correctly classified instances,
    and misclassified instances per category. The results are returned as a DataFrame
    sorted by total instances per category in descending order.

    Parameters:
    - y_test (array-like): True labels of the test set.
    - y_pred (array-like): Predicted labels of the test set.
    - y_multi_class (array-like): Corresponding attack categories of each instance in the test set.

    Returns:
    - pd.DataFrame: A DataFrame containing the following columns:
        - 'category': Unique attack categories.
        - 'accuracy': Accuracy of predictions within each category.
        - 'total_instances': Total number of instances per category.
        - 'correctly_classified': Number of instances correctly classified per category.
        - 'misclassified': Number of instances misclassified per category.
      The DataFrame is sorted by 'total_instances' in descending order.
    """
    df = pd.DataFrame({
        'attackCategory': y_multi_class.tolist(),
        'label': y_test.tolist(),
        'pred': y_pred.tolist()
    })

    unique_categories = df['attackCategory'].unique()
    
    results = []
    for category in unique_categories:
        # Filter rows for current category
        df_filtered = df[df['attackCategory'] == category]
        
        # Calculate accuracy
        accuracy = (df_filtered['label'] == df_filtered['pred']).mean()
        
        # Count total instances
        total_instances = len(df_filtered)

        # Count correctly classified instances
        correctly_classified = sum(df_filtered['label'] == df_filtered['pred'])

        # Count misclassified instances
        misclassified = total_instances - correctly_classified

        # Append to results
        results.append({
            'category': category,
            'accuracy': accuracy,
            'total_instances': total_instances,
            'correctly_classified': correctly_classified,
            'misclassified': misclassified,
        })

    # Convert to DataFrame for easier visualization
    df_attackwise_accuracy = pd.DataFrame(results)

    # Sort by category in descending order
    df_attackwise_accuracy.sort_values('category', ascending=False, inplace=True)
    df_attackwise_accuracy.reset_index(drop=True, inplace=True)
    return df_attackwise_accuracy

def save_to_pickle(data, filename):
    """
    Save a Python object to a pickle file.
    
    Parameters:
    - data: The Python object to be saved.
    - filename: The path and name of the file where the object will be saved.
    """
    with open(filename, 'wb') as file:
        pickle.dump(data, file)


def load_from_pickle(filename):
    """
    Load a Python object from a pickle file.
    
    Parameters:
    - filename: The path and name of the file from which the object will be loaded.
    
    Returns:
    - The loaded Python object.
    """
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def configure_gpu_xgb(device):
    """
    Configure the GPU settings for XGBoost.
    """
    if "cuda" in device:
        gpu_id = int(device.split(":")[1]) if ":" in device else 0
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        return "hist", "cuda"
    else:
        return "hist", "cpu"

def fix_seeds(seed=42):
    """
    Fix random seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def disagreement(predictions_1, predictions_2):

    assert len(predictions_1) == len(predictions_2), "The number of predictions must be the same for both classifiers."
    
    N = len(predictions_1)
    disagree_count = sum(p1 != p2 for p1, p2 in zip(predictions_1, predictions_2))
    
    return disagree_count / N

def ensemble_disagreement(predictions):
    classifier_pairs = combinations(predictions.keys(), 2)
    total_disagreement = 0
    pair_count = 0

    for classifier1, classifier2 in classifier_pairs:
        pred1, pred2 = predictions[classifier1], predictions[classifier2]
        
        disagreement_value = disagreement(pred1, pred2)

        total_disagreement += disagreement_value
        pair_count += 1

    return total_disagreement / pair_count if pair_count > 0 else 0

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

class ParetoAnalysis:
    def __init__(self, res, deduplicate=True):
        """
        Initialize the ParetoAnalysis with a pymoo result object.
        
        Parameters
        ----------
        res : pymoo result
            The result from calling pymoo's minimize(...) with save_history=True.
        deduplicate : bool
            If True, solutions across all generations with the same X 
            will be collapsed to the first occurrence.
        """
        self.res = res
        self.deduplicate = deduplicate
        
        # 1) Gather all solutions from all generations
        self._collect_all_solutions()
        
        # 2) Optionally deduplicate the solutions
        if self.deduplicate:
            self._deduplicate_solutions()
        
        # 3) Compute global PF (across all gens)
        self._compute_global_pf()
        
        # 4) Compute final PF (last gen only)
        self._compute_final_pf()

    def _collect_all_solutions(self):
        """
        Collect all solutions from all generations, storing:
          - all_X (decision variables)
          - all_F (objective values)
          - all_G (generation index)
          - all_I (index within that generation)
        """
        all_X, all_F, all_G, all_I = [], [], [], []
        
        # Enumerate over all generations
        for gen_idx, algo in enumerate(self.res.history):
            pop = algo.pop
            X_gen = pop.get("X")
            F_gen = pop.get("F")

            # Store each individual
            for i_idx, (x_sol, f_sol) in enumerate(zip(X_gen, F_gen)):
                all_X.append(x_sol)
                all_F.append(f_sol)
                all_G.append(gen_idx)
                all_I.append(i_idx)

        # Convert to numpy arrays
        self.all_X = np.array(all_X)  # shape: (N, n_vars)
        self.all_F = np.array(all_F)  # shape: (N, n_objs)
        self.all_G = np.array(all_G)  # shape: (N,)
        self.all_I = np.array(all_I)  # shape: (N,)

    def _deduplicate_solutions(self):
        """
        Remove duplicate solutions based on X, keeping only the *first* occurrence
        (i.e., from the earliest generation).
        
        By default, requires an *exact* match of X to consider it a duplicate.
        """
        # Number of decision variables and objectives
        n_var = self.all_X.shape[1] if self.all_X.ndim > 1 else 1
        n_obj = self.all_F.shape[1] if self.all_F.ndim > 1 else 1

        # Build a DataFrame for easy deduplication
        # We create columns for X_0, X_1, ..., plus F_0, F_1, ..., G, I
        columns = {}
        for v in range(n_var):
            columns[f"X_{v}"] = self.all_X[:, v] if n_var > 1 else self.all_X[:]
        for o in range(n_obj):
            columns[f"F_{o}"] = self.all_F[:, o] if n_obj > 1 else self.all_F[:]

        columns["Gen"] = self.all_G
        columns["Ind"] = self.all_I

        df = pd.DataFrame(columns)

        # List of column names that define "duplicates" (here, just the X columns)
        x_cols = [f"X_{v}" for v in range(n_var)]

        # Sort by generation so that the earliest generation appears first
        df.sort_values(by=["Gen", "Ind"], inplace=True)

        # Drop duplicates, keeping the *first* occurrence (lowest generation).
        df.drop_duplicates(subset=x_cols, keep="first", inplace=True)

        # Sort back by original index if you like, or just keep as is
        # df.sort_index(inplace=True)

        # Convert back to numpy
        # Because df might be in different order, we re-extract in the current order
        unique_X = []
        unique_F = []
        unique_G = df["Gen"].values
        unique_I = df["Ind"].values

        # Rebuild X and F from the dataframe columns
        # same dimension as original
        for idx in df.index:
            row = df.loc[idx]
            x_vals = [row[f"X_{v}"] for v in range(n_var)]
            f_vals = [row[f"F_{o}"] for o in range(n_obj)]
            unique_X.append(x_vals)
            unique_F.append(f_vals)

        self.all_X = np.array(unique_X)
        self.all_F = np.array(unique_F)
        self.all_G = np.array(unique_G)
        self.all_I = np.array(unique_I)

    def _compute_global_pf(self):
        """
        Perform a non-dominated sort on all (unique) solutions (across all generations)
        to get the 'global' Pareto front.
        """
        nds = NonDominatedSorting()
        front_idx = nds.do(self.all_F, only_non_dominated_front=True)

        self.global_pf_X = self.all_X[front_idx]
        self.global_pf_F = self.all_F[front_idx]
        self.global_pf_G = self.all_G[front_idx]
        self.global_pf_I = self.all_I[front_idx]

    def _compute_final_pf(self):
        """
        Extract the final population from the last generation (res.history[-1]),
        then find which ones are non-dominated in that generation alone.
        """
        final_algo = self.res.history[-1]  # last generation
        final_X = final_algo.pop.get("X")
        final_F = final_algo.pop.get("F")

        # Mark the generation index and individual index
        final_gen_idx = len(self.res.history) - 1
        final_G = np.array([final_gen_idx] * len(final_X))
        final_I = np.arange(len(final_X))

        # Non-dominated sort in the final generation
        nds = NonDominatedSorting()
        front_idx_final = nds.do(final_F, only_non_dominated_front=True)

        self.final_pf_X = final_X[front_idx_final]
        self.final_pf_F = final_F[front_idx_final]
        self.final_pf_G = final_G[front_idx_final]
        self.final_pf_I = final_I[front_idx_final]

    def get_all_solutions(self):
        """
        Return all solutions across all generations (after any deduplication).
        
        Returns
        -------
        (all_X, all_F, all_G, all_I)
        """
        return self.all_X, self.all_F, self.all_G, self.all_I

    def get_global_pf(self):
        """
        Return the global (across all generations) Pareto front (X, F, G, I).
        """
        return self.global_pf_X, self.global_pf_F, self.global_pf_G, self.global_pf_I

    def get_final_pf(self):
        """
        Return the final generation's non-dominated solutions (X, F, G, I).
        """
        return self.final_pf_X, self.final_pf_F, self.final_pf_G, self.final_pf_I

def majority_voting(group, pred_col='predictions'):
    """Returns the most common prediction in the group (hard voting). 
       If there's a tie, classify as attack (1) by default.
    """
    mode_values = group[pred_col].mode()
    if len(mode_values) > 1:  # Tie case
        return 1  # Default to attack
    return mode_values[0]

def any_voting(group, pred_col='predictions'):
    """Returns 1 (attack) if at least one classifier predicted 1, otherwise returns 0."""
    return 1 if (group[pred_col] == 1).any() else 0

def soft_voting(group, prob0_col='prob_class_0', prob1_col='prob_class_1'):
    """Computes the average probability for each class and determines the prediction (soft voting)."""
    avg_prob_class_0 = group[prob0_col].mean()
    avg_prob_class_1 = group[prob1_col].mean()
    final_prediction = 0 if avg_prob_class_0 > avg_prob_class_1 else 1
    return pd.Series({'final_prediction_soft': final_prediction, 
                      'avg_prob_class_0': avg_prob_class_0, 
                      'avg_prob_class_1': avg_prob_class_1})
