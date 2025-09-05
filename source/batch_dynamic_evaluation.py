"""
batch_dynamic_evaluation.py

This script automatically discovers CSV files from dynamic selection (Step 6) 
and runs "dynamic_selection_evaluation.py" on each file to generate performance metrics. 
It extracts pool, metric, and aggregation info from each filename, then saves the results 
in a dedicated "metrics" folder for later analysis.

Requires:
  - dynamic_selection_evaluation.py in the same directory or in PATH.
  - A directory containing CSV files from dynamic_selection.py in ../results/classification/.
"""

import os
import subprocess
import argparse
from icecream import ic

if __name__ == "__main__":
  # Set up the command line arguments
  parser = argparse.ArgumentParser(description='Batch dynamic selection evaluation')
  parser.add_argument('--input', type=str, default='../results/classification/', help='Input directory containing CSV files')
  parser.add_argument('--output', type=str, default='../results/classification/metrics/', help='Output directory to save metrics')
  parser.add_argument('--skip-exist', action='store_true', help='If set, skip metrics computation when the output file already exists.')
  parser.add_argument('--generalization', action='store_true', help='Use generalization set instead of test set.')
  parser.add_argument('--normalize', action='store_true', help='Normalize the data before computing distances.')
  parser.add_argument('--constraints', action='store_true', help='Use selected models based on constraints.')
  args = parser.parse_args()

  input_path_dir = args.input
  output_path_dir = args.output

  if args.generalization:
    input_path_dir = os.path.join(input_path_dir, 'generalization')
    output_path_dir = os.path.join(output_path_dir, 'generalization')
  if args.normalize:
    input_path_dir = os.path.join(input_path_dir, 'normalize')
    output_path_dir = os.path.join(output_path_dir, 'normalize')
  if args.constraints:
    input_path_dir = os.path.join(input_path_dir, 'constraints')
    output_path_dir = os.path.join(output_path_dir, 'constraints')

  # Create the output directory if it doesn't exist
  os.makedirs(output_path_dir, exist_ok=True)

  # Iterate over each CSV file in ../results/classification/
  for filename in os.listdir(input_path_dir):
      if filename.endswith(".csv"):
          # Print the filename being processed

          # Extract identifiers (pool, metric, etc.) from the filename
          parts = filename.split('_')
          # Example filename structure: classification_results_<pool>_<metric>_<accuracy>_<something>.csv
          pool = parts[2]
          similarity_metric = parts[3]
          aggregation_method = parts[5].split('.')[0]

          # Build full paths for input and output
          input_file = os.path.join(input_path_dir, filename)
          output_file = os.path.join(output_path_dir, filename)

          # Prepare the output paths
          dir_part, filename = os.path.split(output_file)
          overall_path = os.path.join(dir_part, f"overall_metrics_{filename}")
          attackwise_path = os.path.join(dir_part, f"attackwise_metrics_{filename}")        

          # Check if the output file already exists
          if args.skip_exist and os.path.exists(overall_path) and os.path.exists(attackwise_path):
              print(f"Skipping {filename} as the output file already exists.")
              continue
          # Print the command to be executed
          print(f"Processing {filename}")

          # Prepare the command to run the evaluation script
          command = [
              "python", "dynamic_selection_evaluation.py",
              "--input", input_file,
              "--output", output_file
          ]

          if args.generalization:
            command.append("--generalization")

          # Run the evaluation script
          subprocess.run(command)