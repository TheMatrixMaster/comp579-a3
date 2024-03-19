#!/bin/bash
#SBATCH --partition=long
#SBATCH --cpus-per-task=18
#SBATCH --mem=8G
#SBATCH --time=12:00:00

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/mila/s/stephen.lu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/mila/s/stephen.lu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/mila/s/stephen.lu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/mila/s/stephen.lu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# Activate conda environment
module --force purge
conda activate ml

python runner.py
