#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Initialize counters
total_tests=0
successful_tests=0

# Function to run a test
run_test() {
    echo "Running test: $@"
    python src/train.py "$@"
    if [ $? -eq 0 ]; then
        echo "Test passed"
        ((successful_tests++))
    else
        echo "Test failed"
    fi
    ((total_tests++))
}

echo "Running existing unit and integration tests..."
python -m pytest src/tests/test_neural_network.py -v
if [ $? -eq 0 ]; then
    echo "Pytest tests passed"
    ((successful_tests++))
else
    echo "Pytest tests failed"
fi
((total_tests++))

echo "Running unit tests for loss functions..."
for loss in cross_entropy mse; do
    run_test --loss $loss --epochs 1 --batch_size 32
done

echo "Running unit tests for optimizers..."
for optimizer in sgd momentum nag rmsprop; do
    run_test --optimizer $optimizer --epochs 1 --batch_size 32
done

echo "Running unit tests for weight decay..."
for weight_decay in 0 0.0001 0.01; do
    run_test --weight_decay $weight_decay --epochs 1 --batch_size 32
done

echo "Running unit tests for activations..."
for activation in relu sigmoid tanh; do
    run_test --activation $activation --epochs 1 --batch_size 32
done

echo "Running unit tests for weight initialization..."
for weight_init in random xavier zero; do
    run_test --weight_init $weight_init --epochs 1 --batch_size 32
done

echo "Running unit tests for learning rates..."
for lr in 0.001 0.01 0.1 1; do
    run_test --learning_rate $lr --epochs 1 --batch_size 32
done

echo "Running unit tests for epochs..."
for epochs in 1 10 50 100; do
    run_test --epochs $epochs --batch_size 32
done

echo "Running unit tests for number of hidden layers..."
for num_hidden in {1..6}; do
    arch="784"
    for ((i=1; i<=num_hidden; i++)); do
        arch="${arch},128"
    done
    arch="${arch},10"
    run_test --network $arch --epochs 1 --batch_size 32
done

echo "Running unit tests for neurons per layer..."
for neurons in 16 32 64 128; do
    arch="784,${neurons},64,10"
    run_test --network $arch --epochs 1 --batch_size 32
done

echo "Running integration tests..."
# Integration test 1: Full training with MNIST
run_test --dataset mnist --epochs 10 --batch_size 32

# Integration test 2: Full training with Fashion-MNIST
run_test --dataset fashion_mnist --epochs 10 --batch_size 32

# Integration test 3: Overfitting check with small dataset (simulate with small epochs and check)
# For simplicity, just run with more epochs on MNIST
run_test --dataset mnist --epochs 50 --batch_size 32

echo "Total tests run: $total_tests"
echo "Successful tests: $successful_tests"
