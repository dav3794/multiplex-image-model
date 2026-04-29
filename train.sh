#!/bin/bash
#SBATCH --partition=common
#SBATCH --qos=mzmyslowski
#SBATCH --nodelist=szary
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --job-name=train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

set -e

if [ -z "$1" ]; then
    echo "Usage: sbatch train.sh <config_file> [gp]"
    echo "  config_file: path to YAML config"
    echo "  gp: pass 'gp' as second arg to use GP training script"
    exit 1
fi

config_file=$1
use_gp=${2:-""}

mkdir -p logs

# Set COMET_API_KEY in your environment or ~/.bashrc before submitting
# export COMET_API_KEY=your_key_here

source ~/venv/bin/activate

if [ "$use_gp" = "gp" ]; then
    echo "Starting GP training with config: $config_file"
    python3 train_masked_model_gp.py "$config_file"
elif [ "$use_gp" = "learnmask" ]; then
    echo "Starting learnmask training with config: $config_file"
    python3 train_masked_model_learnmask.py "$config_file"
else
    echo "Starting standard training with config: $config_file"
    python3 train_masked_model.py "$config_file"
fi
