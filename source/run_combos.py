"""
run_combos.py
-------------
Loads chosen 4-of-16 class combos from crs_chosen_combos.pkl and iteratively
calls NSGA2Proposal.py with each combo as the --holdout_attacks argument.
"""

import subprocess
import pickle
import time

if __name__ == "__main__":
    # Start timing the script execution
    start_time = time.time()

    # 1) Load the chosen combos from the pickle file
    with open("../results/crs/crs_chosen_combos.pkl", "rb") as f:
        chosen_combos = pickle.load(f)

    # 2) For each combination, call NSGA2Proposal.py with appropriate args
    for idx, combo in enumerate(chosen_combos, start=1):
        # Convert the tuple (e.g. (0,1,2,3)) into a string representation
        # that your main script can parse. For example: "0,1,2,3"
        combo_str = ",".join(map(str, combo))

        command = [
            "python", "NSGA2Proposal.py",
            "--exec_name", "MultiModalOS-IDS_" + str(idx),
            "--cores", "50",
            "--metric", "auc",
            "--device", "cuda:2",
            #"--checkpoint", "some_checkpoint_file",  # adapt if needed
            "--holdout_attacks", combo_str  # pass the classes to exclude
        ]
        print(f"\n=== Running with combo: {combo} - {idx} ===")
        subprocess.run(command)
    # End timing the script execution
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print elapsed time in seconds
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    # Print elapsed time in minutes
    print(f"Elapsed time: {elapsed_time / 60:.2f} minutes")
