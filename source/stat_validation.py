"""
Statistical validation helper (defaults to the reported generalization holdout).

Computes bootstrap summaries and paired tests for DIME-IDS vs baseline classifiers
using existing outputs:
- results/classification/classification_results_1_euclidean_accuracy_soft.csv
- results/baseline/generalization/baseline_results_pool_1*.pkl

Outputs metrics to stdout: mean±sd, 95% CI, per-attack FN, Wilcoxon (FN), McNemar.
"""

import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import wilcoxon, binomtest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


def load_dime_predictions(classification_csv: str):
    """Aggregate raw classification results to per-query soft predictions."""
    from dynamic_selection_evaluation import (
        aggregate_individual_votes,
        aggregate_query_votes,
    )

    df_cls = pd.read_csv(classification_csv)
    df_ind = aggregate_individual_votes(df_cls)
    df_qr = aggregate_query_votes(df_ind)
    y_true = df_qr["y_true"].to_numpy(dtype=int)
    prob = df_qr["avg_prob_class_1"].to_numpy(dtype=float)
    labels = (prob >= 0.5).astype(int)
    idx = df_qr["query_index"].to_numpy(dtype=int)
    return idx, y_true, prob, labels


def load_baseline_predictions(dataset_csv: str, features: list, idx, baseline_pkl: str):
    """Load baseline model from pkl and predict on provided indices."""
    full = pd.read_csv(dataset_csv)
    with open(baseline_pkl, "rb") as f:
        data = pickle.load(f)
    clf = data["all_features"]["clf"]
    X = full.loc[idx, features]
    prob = clf.predict_proba(X)[:, 1]
    labels = (prob >= 0.5).astype(int)
    return prob, labels


def compute_metrics(y_true, prob, labels):
    acc = accuracy_score(y_true, labels)
    f1 = f1_score(y_true, labels)
    auc = roc_auc_score(y_true, prob)
    cm = confusion_matrix(y_true, labels, labels=[0, 1])
    fn_rate = cm[1, 0] / cm[1].sum()
    return acc, f1, auc, fn_rate


def bootstrap_metrics(y_true, prob, labels, B=2000, rng=None):
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(y_true)
    acc_bs, f1_bs, auc_bs, fn_bs = [], [], [], []
    for _ in range(B):
        idxs = rng.integers(0, n, size=n)
        yt = y_true[idxs]
        p = prob[idxs]
        lab = labels[idxs]
        cm = confusion_matrix(yt, lab, labels=[0, 1])
        fn = cm[1, 0] / cm[1].sum()
        acc_bs.append(accuracy_score(yt, lab))
        f1_bs.append(f1_score(yt, lab))
        auc_bs.append(roc_auc_score(yt, p))
        fn_bs.append(fn)
    return {
        "acc": summary(acc_bs),
        "f1": summary(f1_bs),
        "auc": summary(auc_bs),
        "fn": summary(fn_bs),
    }


def summary(arr):
    arr = np.asarray(arr, dtype=float)
    mean = float(np.mean(arr))
    sd = float(np.std(arr, ddof=1))
    ci = np.percentile(arr, [2.5, 97.5]).tolist()
    return mean, sd, ci


def per_attack_fn(y_true, labels, attacks):
    attack_names = np.unique(attacks[attacks != "normal"])
    fn_list = []
    for a in attack_names:
        m = attacks == a
        yt = y_true[m]
        lab = labels[m]
        cm = confusion_matrix(yt, lab, labels=[0, 1])
        fn = cm[1, 0] / cm[1].sum()
        fn_list.append(fn)
    return attack_names, fn_list


def mcnemar_p(y_true, labels_a, labels_b):
    a_correct = labels_a == y_true
    b_correct = labels_b == y_true
    ct = Counter(zip(a_correct, b_correct))
    # contingency: [[both correct, a correct/b wrong], [a wrong/b correct, both wrong]]
    cont = [
        [ct[(True, True)], ct[(True, False)]],
        [ct[(False, True)], ct[(False, False)]],
    ]
    n01 = cont[0][1]
    n10 = cont[1][0]
    if n01 + n10 == 0:
        p = 1.0
    else:
        p = binomtest(k=min(n01, n10), n=n01 + n10, p=0.5, alternative="two-sided").pvalue
    return cont, p


def main():
    parser = argparse.ArgumentParser(description="Statistical validation helper")
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    parser.add_argument(
        "--dataset",
        default=os.path.join(repo_root, "dataset", "MultiModalOS-IDS.csv"),
        help="Full dataset CSV with features and labels",
    )
    parser.add_argument(
        "--classification",
        default=os.path.join(
            repo_root,
            "results",
            "classification",
            "classification_results_1_euclidean_accuracy_soft.csv",
        ),
        help="DIME per-clf classification results CSV",
    )
    parser.add_argument(
        "--baseline-pkl",
        default=os.path.join(
            repo_root, "results", "baseline", "generalization", "baseline_results_pool_1.pkl"
        ),
        help="Baseline pickle (Global-XGB)",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=2000,
        help="Number of bootstrap resamples",
    )
    args = parser.parse_args()

    sys.path.append(os.path.abspath(os.path.join(repo_root, "source")))
    from EspPipeML import esp_utilities

    features = esp_utilities.DatasetLoader.features

    idx, y_true, prob_d, labels_d = load_dime_predictions(args.classification)
    prob_b, labels_b = load_baseline_predictions(args.dataset, features, idx, args.baseline_pkl)

    m_d = compute_metrics(y_true, prob_d, labels_d)
    m_b = compute_metrics(y_true, prob_b, labels_b)

    rng = np.random.default_rng(0)
    boot_d = bootstrap_metrics(y_true, prob_d, labels_d, B=args.bootstrap, rng=rng)
    boot_b = bootstrap_metrics(y_true, prob_b, labels_b, B=args.bootstrap, rng=rng)

    full = pd.read_csv(args.dataset)
    attacks = full.loc[idx, "classe_atk"].to_numpy()
    _, fn_d = per_attack_fn(y_true, labels_d, attacks)
    _, fn_b = per_attack_fn(y_true, labels_b, attacks)
    stat_w, p_w = wilcoxon(fn_d, fn_b)

    cont, p_m = mcnemar_p(y_true, labels_b, labels_d)

    def fmt(name, metrics, boot):
        acc, f1, auc, fn = metrics
        acc_s = boot["acc"]
        f1_s = boot["f1"]
        auc_s = boot["auc"]
        fn_s = boot["fn"]
        print(f"{name}:")
        print(f"  Acc {acc:.4f} (mean {acc_s[0]:.4f} ± {acc_s[1]:.4f}, CI {acc_s[2]})")
        print(f"  F1  {f1:.4f} (mean {f1_s[0]:.4f} ± {f1_s[1]:.4f}, CI {f1_s[2]})")
        print(f"  AUC {auc:.4f} (mean {auc_s[0]:.4f} ± {auc_s[1]:.4f}, CI {auc_s[2]})")
        print(
            f"  FN% {fn*100:.2f} (mean {fn_s[0]*100:.2f} ± {fn_s[1]*100:.2f}, CI {[round(x*100,2) for x in fn_s[2]]})"
        )

    fmt("DIME-IDS", m_d, boot_d)
    fmt("Global-XGB", m_b, boot_b)
    print(f"Wilcoxon on per-attack FN: p={p_w:.4g}")
    print(f"McNemar contingency {cont}, p={p_m:.4g}")


if __name__ == "__main__":
    main()

#python code/source/stat_validation.py --classification code/results/classification/generalization/normalize/constraints/classification_results_1_euclidean_accuracy_soft.csv --bootstrap 2000