#!/bin/bash

# Run W&B Experiments 2.1-2.12 for DA6401 Assignment 1

echo "========================================"
echo "Setting up environment"
echo "========================================"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "================================"
echo "Running W&B Experiments"
echo "================================"
echo ""

PROJECT_NAME="wandb_run1"

# Use command line args if provided
if [ $# -ge 1 ]; then
    PROJECT_NAME="$1"
fi

COUNT=10
if [ $# -ge 2 ]; then
    COUNT="$2"
fi

# Initialize counters
total_tests=0
successful_tests=0
successful_experiments=0

# Function to run experiment with W&B
run_experiment() {
    name="$1"
    shift
    echo "Running Experiment $name"
    python3 src/train.py \
        --use_wandb \
        --wandb_project "$PROJECT_NAME" \
        "$@"
    echo ""
}

# 2.2 Hyperparameter Sweep Analysis
echo "Experiment 2.2: Hyperparameter Sweep Analysis"
echo "=============================================="

# Clear previous log files
rm -f best_configs.log overfit_configs.log

# Create sweep config
cat > sweep_config.yaml << 'EOFF'
program: src/train.py
method: bayes
metric:
  name: val_acc
  goal: maximize
parameters:
  dataset:
    values: [mnist]
  epochs:
    values: [15, 20, 25]
  batch_size:
    values: [16, 32, 64]
  loss:
    values: [cross_entropy, mse]
  learning_rate:
    values: [0.001, 0.01, 0.1]
  optimizer:
    values: [sgd, momentum, nag, rmsprop]
  weight_decay:
    values: [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
  num_layers:
    min: 1
    max: 6
  hidden_size:
    values: [32, 64, 128]
  activation:
    values: [sigmoid, tanh, relu]
  weight_init:
    values: [random, xavier]
  wandb_project:
    value: "mlp-numpy"
command:
  - ${env}
  - python3
  - ${program}
  - "--use_wandb"
  - ${args}
EOFF

# Update wandb_project in sweep config
sed -i.bak '/wandb_project:/{n;s/    value: ".*"/    value: "'"$PROJECT_NAME"'"/;}' sweep_config.yaml

# Initialize sweep
SWEEP_ID=$(python3 -m wandb sweep sweep_config.yaml --project "$PROJECT_NAME" 2>&1 | grep "wandb agent" | sed 's/.*wandb agent //' | awk '{print $1}')
echo "Sweep ID: $SWEEP_ID"

# Run sweep agent for 10 runs
python3 -m wandb agent $SWEEP_ID --count $COUNT && ((successful_experiments++))

# 2.2 Analysis: fetch runs from sweep, create parallel coordinates, update best_config.json
SWEEP_PROJECT="$PROJECT_NAME"
echo "Running Experiment 2.2 analysis on sweep: $SWEEP_PROJECT"
python3 src/experiment.py --experiment 2.2 --use_wandb --wandb_project "$PROJECT_NAME" --sweep_name "$SWEEP_PROJECT" && ((successful_experiments++))

# 2.7 Overfitting Analysis (runs separately after sweep)
echo "Experiment 2.7: Overfitting Analysis"
python3 src/experiment.py --experiment 2.7 --use_wandb --wandb_project "$PROJECT_NAME" --sweep_name "$SWEEP_PROJECT" && ((successful_experiments++))
echo ""

# 2.1 Data Exploration and Class Distribution
echo "Experiment 2.1: Data Exploration and Class Distribution"
echo "======================================================"
python3 src/experiment.py --experiment 2.1 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.3 The Optimizer Showdown
echo "Experiment 2.3: The Optimizer Showdown"
echo "======================================"
python3 src/experiment.py --experiment 2.3 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.4 Vanishing Gradient Analysis
echo "Experiment 2.4: Vanishing Gradient Analysis"
echo "==========================================="
python3 src/experiment.py --experiment 2.4 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.5 The "Dead Neuron" Investigation
echo "Experiment 2.5: The \"Dead Neuron\" Investigation"
echo "================================================"
python3 src/experiment.py --experiment 2.5 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.6 Loss Function Comparison
echo "Experiment 2.6: Loss Function Comparison"
echo "========================================"
python3 src/experiment.py --experiment 2.6 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.8 Error Analysis
echo "Experiment 2.8: Error Analysis"
echo "=============================="
python3 src/experiment.py --experiment 2.8 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.9 Weight Initialization & Symmetry
echo "Experiment 2.9: Weight Initialization & Symmetry"
echo "================================================"
python3 src/experiment.py --experiment 2.9 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

# 2.10 The Fashion-MNIST Transfer Challenge
echo "Experiment 2.10: The Fashion-MNIST Transfer Challenge"
echo "====================================================="
python3 src/experiment.py --experiment 2.10 --use_wandb --wandb_project "$PROJECT_NAME" && ((successful_experiments++))
echo ""

echo "Successful experiments: $successful_experiments / 10"
echo "All experiments completed. Check W&B dashboard for logs."
