#!/bin/bash
#OAR -l nodes=1/gpu=4,walltime=12:00:00  
#OAR -n llama_3b_training_full 

cd $WORKDIR

git clone git@github.com:cathK1/Llama3.2ToM.git
cd Llama3.2ToM

export DATA_PATH="$WORKDIR/Llama3.2ToM/Llama3.2/3b/data"
export OUTPUT_PATH="$WORKDIR/Llama3.2ToM/Llama3.2/3b/output"

mkdir -p $OUTPUT_PATH

module load python/3.8 

source $WORKDIR/Llama3.2ToM/venv/bin/activate

python Llama3.2/3b/llama_3b.py
