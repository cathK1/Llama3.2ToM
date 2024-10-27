#!/bin/bash
# OAR options for Grid'5000 cluster
#OAR -l nodes=1,walltime=02:00:00  # Specify one node and 2 hours wall time
#OAR -n llama_3b_training_job       # Name of the job

# Set paths for data and output directories
export DATA_PATH="./Llama3.2/3b/data"
export OUTPUT_PATH="./Llama3.2/3b/output"

# Load necessary modules if the cluster requires it (example: loading Python)
module load python/3.8  # Adjust the Python version if necessary

# Activate the virtual environment
source venv/bin/activate

# Run the Python script
python Llama3.2/3b/llama_3b.py

