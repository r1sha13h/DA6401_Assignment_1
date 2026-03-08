"""
Experiment 2.1: Data Exploration and Class Distribution
Log a W&B Table with 5 sample images from each of the 10 classes and their predicted labels.
"""

import argparse
import numpy as np
import wandb
import matplotlib.pyplot as plt
import sklearn.model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import json

# Add src to path
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode

def parse_arguments():
    parser = argparse.ArgumentParser(description='Experiments')
    parser.add_argument('--experiment', type=str, required=True, help='Experiment number')
    parser.add_argument('--use_wandb', action='store_true', help='Use W&B')
    parser.add_argument('--wandb_project', type=str, default='mlp-numpy', help='W&B project name')
    parser.add_argument('--sweep_name', type=str, default='', help='W&B sweep project name to fetch runs from (for 2.2/2.7)')
    args = parser.parse_args()
    return args

def load_best_config():
    """Load the best config from best_config.json (handles both old dict and new list format)."""
    with open('src/best_config.json', 'r') as f:
        data = json.load(f)
    if isinstance(data, list):
        cfg = data[0]['config']
    else:
        cfg = data
    if isinstance(cfg.get('hidden_size'), int):
        cfg['hidden_size'] = [cfg['hidden_size']] * cfg.get('num_layers', 1)
    return cfg

def main():
    args = parse_arguments()

    if args.experiment == '2.1':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.1_Data_Exploration", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')

        # Split train for quick training
        X_train_small, _, y_train_small, _ = train_test_split(X_train, y_train, test_size=0.9, random_state=42)

        # One-hot encode
        y_train_onehot = one_hot_encode(y_train_small)

        # Load best config from 2.2
        best_config = load_best_config()

        # Quick model using best config
        model_args = type('Args', (), best_config)()

        print("Training quick model...")
        model = NeuralNetwork(model_args)
        model.train(X_train_small, y_train_onehot, epochs=model_args.epochs, batch_size=model_args.batch_size)

        # Sample 5 images per class
        table_data = []
        for class_id in range(10):
            indices = np.where(y_test == class_id)[0][:5]
            for idx in indices:
                img_flat = X_test[idx]
                img = img_flat.reshape(28, 28)
                pred = model.predict(img_flat.reshape(1, -1))[0]
                table_data.append([wandb.Image(img), pred])

        # Create table
        table = wandb.Table(data=table_data, columns=["Image", "Predicted Label"])

        # Log to W&B
        if args.use_wandb:
            wandb.log({"Sample Images and Predictions": table})
            print("Logged table to W&B.")
        else:
            print("W&B not used, table not logged.")

        print("Experiment 2.1 completed.")

    elif args.experiment == '2.3':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.3_Optimizer_Comparison", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        optimizers = ['sgd', 'momentum', 'nag', 'rmsprop']
        epochs = 20
        
        lrs = [0.001, 0.01, 0.1]

        for lr in lrs:
            print(f"\nTraining with lr={lr}")
            results = {}

            for opt in optimizers:
                print(f"Training with {opt}, lr={lr}, epochs={epochs}")

                model_args = type('Args', (), {
                    'num_layers': 3,
                    'hidden_size': [128, 128, 128],
                    'activation': 'relu',
                    'weight_init': 'xavier',
                    'loss': 'cross_entropy',
                    'optimizer': opt,
                    'learning_rate': lr,
                    'weight_decay': 0.0
                })()

                model = NeuralNetwork(model_args)
                model.train(X_train, y_train_onehot, epochs=epochs, batch_size=64, val_X=X_val, val_y=y_val_onehot)

                results[opt] = {
                    'val_loss_history': model.val_loss_history,
                    'val_acc_history': model.val_acc_history
                }

            # Create 2 plots for this lr
            epochs_range = list(range(1, epochs + 1))
            colors = ['blue', 'green', 'red', 'cyan', 'magenta']
            lr_str = str(lr).replace('.', '')

            # Plot 1: Validation Loss for this lr
            fig1, ax1 = plt.subplots(figsize=(12, 6))
            for i, opt in enumerate(optimizers):
                ax1.plot(epochs_range, results[opt]['val_loss_history'], label=opt.upper(), color=colors[i], linewidth=2)
            ax1.set_title(f'Validation Loss Comparison (LR={lr})')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            plt.tight_layout()

            # Save plot to local folder
            plt.savefig(os.path.join('plot', f'lr{lr_str}_val_loss.png'))

            if args.use_wandb:
                wandb.log({f"lr{lr_str}_val_loss": wandb.Image(fig1)})
            plt.close(fig1)

            # Plot 2: Validation Accuracy for this lr
            fig2, ax2 = plt.subplots(figsize=(12, 6))
            for i, opt in enumerate(optimizers):
                ax2.plot(epochs_range, results[opt]['val_acc_history'], label=opt.upper(), color=colors[i], linewidth=2)
            ax2.set_title(f'Validation Accuracy Comparison (LR={lr})')
            ax2.set_xlabel('Epochs')
            ax2.set_ylabel('Accuracy')
            ax2.legend()
            ax2.grid(True)
            plt.tight_layout()

            # Save plot to local folder
            plt.savefig(os.path.join('plot', f'lr{lr_str}_val_acc.png'))

            if args.use_wandb:
                wandb.log({f"lr{lr_str}_val_acc": wandb.Image(fig2)})
            plt.close(fig2)

        print("Experiment 2.3 completed.")

    elif args.experiment == '2.4':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.4_Vanishing_Gradient_Analysis", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        activations = ['sigmoid', 'tanh', 'relu']
        num_layers_list = [4, 6, 8, 10, 12]

        for act in activations:
            print(f"Plotting gradient norms for {act}")
            fig, ax = plt.subplots(figsize=(10, 6))
            colors = ['blue', 'green', 'red', 'purple', 'orange']

            for i, num_hidden in enumerate(num_layers_list):
                hidden_size = [128] * num_hidden
                lr = 0.0001 if act == 'sigmoid' else 0.001
                model_args = type('Args', (), {
                    'num_layers': num_hidden,
                    'hidden_size': hidden_size,
                    'activation': act,
                    'weight_init': 'xavier',
                    'loss': 'cross_entropy',
                    'optimizer': 'rmsprop',
                    'learning_rate': lr,
                    'weight_decay': 0.0
                })()

                model = NeuralNetwork(model_args)
                model.train(X_train, y_train_onehot, epochs=15, batch_size=64, val_X=X_val, val_y=y_val_onehot)

                epochs_range = list(range(1, 16))
                ax.plot(epochs_range, model.grad_norms_first_history, label=f'{num_hidden} layers', color=colors[i])

            ax.set_title(f'{act.upper()} Gradient Norms for First Hidden Layer')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Gradient Norm')
            ax.legend()

            plt.tight_layout()

            if args.use_wandb:
                wandb.log({f"{act}_gradient_norms_plot": wandb.Image(plt)})

            plt.close()

        print("Experiment 2.4 completed.")

    elif args.experiment == '2.5':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.5_Dead_Neuron_Investigation", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        activations = ['relu', 'tanh']
        lrs = [0.0001, 0.001, 0.01, 0.1]

        for lr in lrs:
            lr_str = str(lr).replace('.', '')
            for act in activations:
                print(f"Training with {act} and lr={lr}")

                model_args = type('Args', (), {
                    'num_layers': 6,
                    'hidden_size': [128]*6,
                    'activation': act,
                    'weight_init': 'xavier',
                    'loss': 'cross_entropy',
                    'optimizer': 'rmsprop',
                    'learning_rate': lr,
                    'weight_decay': 0.0
                })()

                model = NeuralNetwork(model_args)
                model.train(X_train, y_train_onehot, epochs=20, batch_size=64, val_X=X_val, val_y=y_val_onehot)

                # Plot validation accuracy
                epochs_range = list(range(1, 21))
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(epochs_range, model.val_acc_history, label='Validation Accuracy', color='blue')
                ax.set_title(f'{act.upper()} LR={lr} Validation Accuracy')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Accuracy')
                ax.legend()
                plt.tight_layout()
                if args.use_wandb:
                    wandb.log({f"{lr_str}_{act}_val_accuracy_plot": wandb.Image(plt)})
                plt.close()

                # Create dead neuron matrix based on epochs dead >=5
                matrix = np.zeros((128, 6))
                for l in range(6):
                    for n in range(128):
                        count = sum(1 for epoch_dead in model.dead_neurons_history if epoch_dead[l][n] == 1)
                        matrix[n, l] = 0 if count >= 5 else 1

                # Plot heatmap
                fig, ax = plt.subplots(figsize=(12, 8))
                im = ax.imshow(matrix, cmap='gray', aspect='auto', vmin=0, vmax=1)
                ax.set_title(f'{act.upper()} LR={lr} Dead Neuron Heatmap')
                ax.set_xlabel('Layers')
                ax.set_ylabel('Neurons')
                ax.set_xticks(range(6))
                ax.set_xticklabels([f'Hidden {k+1}' for k in range(6)], rotation=45, ha='right')
                cbar = plt.colorbar(im, ax=ax, ticks=[0, 1])
                cbar.set_ticklabels(['Dead (0)', 'Alive (1)'])
                plt.tight_layout()
                if args.use_wandb:
                    wandb.log({f"{lr_str}_{act}_dead_neuron_heatmap": wandb.Image(plt)})
                plt.close()

        print("Experiment 2.5 completed.")

    elif args.experiment == '2.6':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.6_Loss_Function_Comparison", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        losses = ['mse', 'cross_entropy']
        networks = [
            {'num_layers': 3, 'hidden_size': [32, 32, 32]},
            {'num_layers': 5, 'hidden_size': [128] * 5}
        ]

        for net in networks:
            for loss in losses:
                print(f"Training with {loss} loss on {net['num_layers']}L network")

                model_args = type('Args', (), {
                    'num_layers': net['num_layers'],
                    'hidden_size': net['hidden_size'],
                    'activation': 'sigmoid',
                    'weight_init': 'xavier',
                    'loss': loss,
                    'optimizer': 'rmsprop',
                    'learning_rate': 0.01,
                    'weight_decay': 0.0
                })()

                model = NeuralNetwork(model_args)
                model.train(X_train, y_train_onehot, epochs=25, batch_size=32, val_X=X_val, val_y=y_val_onehot)

                # Compute final test accuracy
                test_pred = model.predict(X_test)
                test_acc = np.mean(test_pred == y_test)

                # Plot accuracies
                fig, ax = plt.subplots(figsize=(10, 6))
                epochs_range = list(range(1, 26))
                ax.plot(epochs_range, model.val_loss_history, label='Val Loss', color='blue')
                ax.plot(epochs_range, model.val_acc_history, label='Val Accuracy', color='green')
                ax.axhline(y=test_acc, color='red', linestyle='--', label=f'Test Accuracy: {test_acc:.4f}')
                ax.set_title(f'{loss.upper()} Loss on {net["num_layers"]}L Network')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss / Accuracy')
                ax.legend()

                plt.tight_layout()

                if args.use_wandb:
                    wandb.log({f"{loss}_accuracy_plot_{net['num_layers']}L": wandb.Image(plt)})

                plt.close()

        print("Experiment 2.6 completed.")

    elif args.experiment == '2.8':
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.8_Error_Analysis", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        # Load best config from 2.2
        best_config = load_best_config()

        # Train model with best config
        model_args = type('Args', (), best_config)()
        model_args.epochs = 20  # Override epochs to 20

        model = NeuralNetwork(model_args)
        model.train(X_train, y_train_onehot, epochs=20, batch_size=64, val_X=X_val, val_y=y_val_onehot)

        # Compute predictions and probabilities
        test_logits = model.forward(X_test)
        test_probs = softmax(test_logits)
        test_pred = np.argmax(test_logits, axis=1)

        # Confusion Matrix
        conf = confusion_matrix(y_test, test_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(conf, cmap='Blues')
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        # Add text annotations
        for i in range(10):
            for j in range(10):
                ax.text(j, i, int(conf[i, j]), ha='center', va='center', color='white' if conf[i, j] > np.max(conf)/2 else 'black')
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"confusion_matrix": wandb.Image(plt)})
        plt.close()

        # Total Samples vs Wrong Predictions
        total_per_class = np.sum(conf, axis=1)
        wrong_per_class = total_per_class - np.diag(conf)
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(10)
        ax.bar(x, total_per_class, color='lightblue', label='Total Samples', width=0.6)
        ax.bar(x, wrong_per_class, color='red', label='Wrong Predictions', width=0.4)
        for i in range(10):
            ax.text(i, total_per_class[i] + 50, f'{int(total_per_class[i])}', ha='center', va='bottom', color='blue')
            ax.text(i, wrong_per_class[i] + 50, f'{int(wrong_per_class[i])}', ha='center', va='bottom', color='red')
        ax.set_title('Total Samples vs Wrong Predictions per Class')
        ax.set_xlabel('Digit Class')
        ax.set_ylabel('Count')
        ax.set_xticks(x)
        ax.legend()
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"total_vs_wrong_predictions": wandb.Image(plt)})
        plt.close()

        # 5 Wrong Predictions
        wrong_indices = np.where(test_pred != y_test)[0][:5]
        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, idx in enumerate(wrong_indices):
            img = X_test[idx].reshape(28, 28)
            axes[i].imshow(img, cmap='gray')
            actual = y_test[idx]
            pred = test_pred[idx]
            prob_actual = test_probs[idx, actual]
            axes[i].set_title(f'Actual: {actual}\nPred: {pred}\nProb Actual: {prob_actual:.2f}')
            axes[i].axis('off')
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"wrong_predictions": wandb.Image(plt)})
        plt.close()

        print("Experiment 2.8 completed.")

    elif args.experiment == '2.9':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.9_Weight_Initialization_Symmetry", group=args.experiment)

        # Load MNIST
        print("Loading MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        # Load best config from 2.2
        best_config = load_best_config()

        inits = ['zeros', 'xavier']
        grad_data = {}

        for init in inits:
            print(f"Training with {init} initialization")

            model_args = type('Args', (), best_config)()
            model_args.weight_init = init
            model_args.epochs = 50

            model = NeuralNetwork(model_args)
            model.train(X_train, y_train_onehot, epochs=50, batch_size=64, val_X=X_val, val_y=y_val_onehot)

            grad_data[init] = model.grad_history_epoch

            # Plot validation accuracy
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs_range = list(range(1, 51))
            ax.plot(epochs_range, model.val_acc_history, label='Validation Accuracy')
            ax.set_title(f'{init.upper()} Init Validation Accuracy')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Accuracy')
            ax.legend()
            plt.tight_layout()
            if args.use_wandb:
                wandb.log({f"{init}_val_accuracy_plot": wandb.Image(plt)})
            plt.close()

            # Plot validation loss
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(epochs_range, model.val_loss_history, label='Validation Loss')
            ax.set_title(f'{init.upper()} Init Validation Loss')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Loss')
            ax.legend()
            plt.tight_layout()
            if args.use_wandb:
                wandb.log({f"{init}_val_loss_plot": wandb.Image(plt)})
            plt.close()

        # Plot gradient norms for each init method
        for init in inits:
            fig, ax = plt.subplots(figsize=(10, 6))
            epochs_range = list(range(1, 51))
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            for i in range(5):
                ax.plot(epochs_range, [h[i] for h in grad_data[init]], label=f'Neuron {i+1}', color=colors[i])
            ax.set_title(f'{init.upper()} Init Gradient Norms of 5 Neurons in First Hidden Layer')
            ax.set_xlabel('Epochs')
            ax.set_ylabel('Gradient Norm')
            ax.legend()
            plt.tight_layout()
            if args.use_wandb:
                wandb.log({f"{init}_gradient_plot": wandb.Image(plt)})
            plt.close()

        print("Experiment 2.9 completed.")

    elif args.experiment == '2.10':
        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.10_Fashion_MNIST_Transfer_Challenge", group=args.experiment)

        # Load Fashion-MNIST
        print("Loading Fashion-MNIST dataset...")
        (X_train, y_train), (X_test, y_test) = load_data('fashion_mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        # Load top 3 configs from best_config.json
        with open('src/best_config.json', 'r') as f:
            data = json.load(f)
        if isinstance(data, list):
            top3_entries = data[:3]
        else:
            top3_entries = [{'val_acc': 0, 'config': data}]

        results = []
        for idx, entry in enumerate(top3_entries):
            cfg = entry['config'] if isinstance(entry, dict) and 'config' in entry else entry
            if isinstance(cfg.get('hidden_size'), int):
                cfg['hidden_size'] = [cfg['hidden_size']] * cfg.get('num_layers', 1)
            cfg_name = f"Config{idx+1}: opt={cfg.get('optimizer','?')} lr={cfg.get('learning_rate','?')} act={cfg.get('activation','?')} layers={cfg.get('num_layers','?')}"
            print(f"\nTraining {cfg_name}")

            model_args = type('Args', (), {
                'num_layers': cfg.get('num_layers', 2),
                'hidden_size': cfg['hidden_size'],
                'activation': cfg.get('activation', 'relu'),
                'weight_init': cfg.get('weight_init', 'xavier'),
                'loss': cfg.get('loss', 'cross_entropy'),
                'optimizer': cfg.get('optimizer', 'sgd'),
                'learning_rate': cfg.get('learning_rate', 0.01),
                'weight_decay': cfg.get('weight_decay', 0.0)
            })()

            n_epochs = cfg.get('epochs', 15)
            model = NeuralNetwork(model_args)
            model.train(X_train, y_train_onehot, epochs=n_epochs, batch_size=cfg.get('batch_size', 64), val_X=X_val, val_y=y_val_onehot)

            # Compute test accuracy
            test_pred = model.predict(X_test)
            test_acc = np.mean(test_pred == y_test)

            # Plot train_acc, val_acc, val_loss per epoch in a single plot
            epochs_range = list(range(1, n_epochs + 1))
            fig, ax1 = plt.subplots(figsize=(12, 7))

            # Left y-axis: accuracies
            ax1.plot(epochs_range, model.train_acc_history, 'r-o', linewidth=2, markersize=3, label='Train Acc')
            ax1.plot(epochs_range, model.val_acc_history, 'g-o', linewidth=2, markersize=3, label='Val Acc')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)

            # Right y-axis: val loss
            ax2 = ax1.twinx()
            ax2.plot(epochs_range, model.val_loss_history, 'b-s', linewidth=2, markersize=3, label='Val Loss')
            ax2.set_ylabel('Validation Loss')

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            # Visible title with full config details (wrapped)
            title_text = (f"{cfg_name}\nTest Acc: {test_acc:.4f}")
            fig.suptitle(title_text, fontsize=11, fontweight='bold', y=0.98)
            plt.tight_layout(rect=[0, 0, 1, 0.92])
            if args.use_wandb:
                wandb.log({f"config{idx+1}_training_curves": wandb.Image(fig)})
            plt.close(fig)

            results.append({
                'name': cfg_name,
                'train_acc': model.train_acc_history[-1],
                'val_acc': model.val_acc_history[-1],
                'test_acc': test_acc
            })

        # Summary bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        labels = [f"Config {i+1}" for i in range(len(results))]
        train_accs = [r['train_acc'] for r in results]
        val_accs = [r['val_acc'] for r in results]
        test_accs = [r['test_acc'] for r in results]

        x = np.arange(len(labels))
        width = 0.25
        ax.bar(x - width, train_accs, width, label='Train Accuracy')
        ax.bar(x, val_accs, width, label='Val Accuracy')
        ax.bar(x + width, test_accs, width, label='Test Accuracy')
        ax.set_title('Top 3 MNIST Configs on Fashion-MNIST')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Accuracy')
        ax.legend()
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"accuracies_summary": wandb.Image(plt)})
        plt.close()

        print("Experiment 2.10 completed.")

    elif args.experiment == '2.2':
        import pandas as pd

        sweep_project = args.sweep_name
        if not sweep_project:
            print("ERROR: --sweep_name is required for experiment 2.2")
            print("Usage: python3 src/experiment.py --experiment 2.2 --use_wandb --wandb_project <analysis_project> --sweep_name <sweep_project>")
            return

        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.2_Hyperparameter_Sweep_Analysis", group=args.experiment)

        api = wandb.Api()
        runs = list(api.runs(f'rishabh-mishra-cer16-iitmaana/{sweep_project}'))
        print(f"Total runs fetched from sweep '{sweep_project}': {len(runs)}")

        # Process each run
        valid_runs = []
        for run in runs:
            try:
                history = run.history()
                if 'val_acc' not in history.columns:
                    print(f"Run {run.name} missing val_acc, skipping")
                    continue
                max_val_acc = history['val_acc'].max()
                if np.isnan(max_val_acc):
                    continue
                valid_runs.append({
                    'name': run.name,
                    'config': run.config,
                    'best_val_acc': max_val_acc,
                    'history': history,
                    'summary': dict(run.summary),
                })
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
                continue

        print(f"Valid runs with val_acc: {len(valid_runs)}")

        if len(valid_runs) == 0:
            print("ERROR: No valid sweep runs found.")
            if args.use_wandb:
                wandb.finish()
            return

        valid_runs.sort(key=lambda r: r['best_val_acc'], reverse=True)

        # --- Per-run plots: combined train_acc, test_acc, val_loss in a single plot ---
        os.makedirs('src/plots', exist_ok=True)
        for rd in valid_runs:
            history = rd['history']
            # Filter to epoch-level rows only (drop rows where val_acc is NaN)
            if 'val_acc' in history.columns:
                history = history.dropna(subset=['val_acc']).reset_index(drop=True)
            config = rd['config']
            epochs = list(range(1, len(history) + 1))
            config_parts = [
                f"opt={config.get('optimizer','?')}",
                f"lr={config.get('learning_rate','?')}",
                f"act={config.get('activation','?')}",
                f"layers={config.get('num_layers','?')}",
                f"hs={config.get('hidden_size','?')}",
                f"bs={config.get('batch_size','?')}",
                f"loss={config.get('loss','?')}",
                f"wd={config.get('weight_decay','?')}",
                f"init={config.get('weight_init','?')}",
            ]
            plot_title = " | ".join(config_parts)

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Left y-axis: accuracies
            if 'train_acc' in history.columns:
                ax1.plot(epochs, history['train_acc'], 'r-', linewidth=2, label='Train Acc')
            final_test_acc = rd['summary'].get('final_test_acc', None)
            if final_test_acc is not None:
                ax1.axhline(y=final_test_acc, color='g', linestyle='--', linewidth=2, label=f'Test Acc: {final_test_acc:.4f}')
            if 'val_acc' in history.columns:
                ax1.plot(epochs, history['val_acc'], 'm-', linewidth=2, label='Val Acc')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)

            # Right y-axis: val loss
            ax2 = ax1.twinx()
            if 'val_loss' in history.columns:
                ax2.plot(epochs, history['val_loss'], 'b-', linewidth=2, label='Val Loss')
            ax2.set_ylabel('Validation Loss')

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            fig.suptitle(plot_title, fontsize=9, y=1.02)
            plt.tight_layout()

            # Save plot locally
            fname = (f"opt-{config.get('optimizer','x')}_lr-{config.get('learning_rate','x')}"
                     f"_act-{config.get('activation','x')}_layers-{config.get('num_layers','x')}"
                     f"_hs-{config.get('hidden_size','x')}_bs-{config.get('batch_size','x')}"
                     f"_loss-{config.get('loss','x')}_wd-{config.get('weight_decay','x')}"
                     f"_init-{config.get('weight_init','x')}.png")
            plt.savefig(os.path.join('src', 'plots', fname), bbox_inches='tight', dpi=150)

            if args.use_wandb:
                wandb.log({f"run_{rd['name']}_curves": wandb.Image(fig)})
            plt.close(fig)

        # --- Parallel Coordinates Plot with train_acc, test_acc, val_loss ---
        print("\nCreating Parallel Coordinates plot...")
        hyperparams = ['epochs', 'batch_size', 'learning_rate', 'num_layers', 'hidden_size', 'weight_decay']
        cat_hyperparams = ['optimizer', 'activation', 'weight_init']
        all_dims = hyperparams + cat_hyperparams
        param_data = []

        for rd in valid_runs:
            config = rd['config']
            row = {
                'train_acc': rd['summary'].get('final_train_acc', rd['summary'].get('train_acc', 0)),
                'test_acc': rd['summary'].get('final_test_acc', 0),
                'val_loss': rd['history']['val_loss'].iloc[-1] if 'val_loss' in rd['history'].columns else 0,
                'val_acc': rd['best_val_acc'],
            }
            for hp in hyperparams:
                val = config.get(hp, 0)
                if isinstance(val, list):
                    val = val[0] if val else 0
                row[hp] = val
            for hp in cat_hyperparams:
                row[hp] = config.get(hp, 'unknown')
            param_data.append(row)

        df = pd.DataFrame(param_data)

        # Encode categoricals for plotting
        df_plot = df.copy()
        for col in cat_hyperparams:
            if col in df_plot.columns:
                df_plot[col] = df_plot[col].astype('category').cat.codes

        plot_cols = all_dims + ['train_acc', 'test_acc', 'val_loss']
        df_norm = df_plot.copy()
        for col in plot_cols:
            if col in df_norm.columns:
                cmin, cmax = df_norm[col].min(), df_norm[col].max()
                if cmax != cmin:
                    df_norm[col] = (df_norm[col] - cmin) / (cmax - cmin)
                else:
                    df_norm[col] = 0.5

        fig, ax = plt.subplots(figsize=(16, 8))
        colors = plt.cm.viridis((df_plot['val_acc'] - df_plot['val_acc'].min()) / (df_plot['val_acc'].max() - df_plot['val_acc'].min() + 1e-8))

        for i in range(len(df_norm)):
            y_vals = [df_norm.iloc[i][p] for p in plot_cols]
            ax.plot(range(len(plot_cols)), y_vals, color=colors[i], alpha=0.6, linewidth=1.5)

        ax.set_xticks(range(len(plot_cols)))
        ax.set_xticklabels(plot_cols, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Parallel Coordinates: Hyperparams + train_acc, test_acc, val_loss (colored by val_acc)')
        ax.grid(True, alpha=0.3)

        sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=df_plot['val_acc'].min(), vmax=df_plot['val_acc'].max()))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Validation Accuracy')
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"parallel_coordinates_plot": wandb.Image(fig)})
        plt.close()

        # Correlation analysis
        numeric_cols = [c for c in hyperparams if c in df_plot.columns]
        correlations = df_plot[numeric_cols + ['val_acc']].corr()['val_acc'].drop('val_acc').abs().sort_values(ascending=False)
        print("\nHyperparameter correlation with Validation Accuracy (absolute):")
        for param, corr in correlations.items():
            print(f"  {param}: {corr:.4f}")
        print(f"\nMost significant hyperparameter: {correlations.index[0]} (correlation: {correlations.iloc[0]:.4f})")
        if args.use_wandb:
            wandb.log({"hyperparam_correlations": dict(correlations)})

        # --- Save this sweep's best configs to best_configs.log ---
        with open('src/best_configs.log', 'w') as f:
            f.write(f"Sweep: {sweep_project}\n")
            f.write("=" * 60 + "\n\n")
            for i, rd in enumerate(valid_runs):
                f.write(f"Rank {i+1}: val_acc {rd['best_val_acc']:.4f}\n")
                f.write(json.dumps(rd['config'], indent=4) + "\n\n")
        print(f"\nSaved {len(valid_runs)} configs to src/best_configs.log")

        # --- Update global best_config.json (top 3 across all sweeps) ---
        existing_entries = []
        if os.path.exists('src/best_config.json'):
            try:
                with open('src/best_config.json', 'r') as f:
                    data = json.load(f)
                if isinstance(data, list):
                    existing_entries = data
                elif isinstance(data, dict):
                    existing_entries = [{'val_acc': data.get('val_acc', 0), 'config': data}]
            except Exception:
                pass

        # Add this sweep's runs
        for rd in valid_runs:
            cfg = rd['config'].copy()
            if isinstance(cfg.get('hidden_size'), int):
                cfg['hidden_size'] = [cfg['hidden_size']] * cfg.get('num_layers', 1)
            # Remove wandb_project from stored config
            cfg.pop('wandb_project', None)
            existing_entries.append({'val_acc': rd['best_val_acc'], 'config': cfg})

        # Deduplicate by config hash, keep best val_acc
        seen = {}
        for entry in existing_entries:
            key = json.dumps(entry['config'], sort_keys=True)
            if key not in seen or entry['val_acc'] > seen[key]['val_acc']:
                seen[key] = entry
        unique_entries = sorted(seen.values(), key=lambda x: x['val_acc'], reverse=True)
        top3 = unique_entries[:3]

        with open('src/best_config.json', 'w') as f:
            json.dump(top3, f, indent=4)
        print(f"Updated src/best_config.json with top 3 global configs (best val_acc: {top3[0]['val_acc']:.4f})")

        # --- Update best_model.npy only if the overall best config changed ---
        best_cfg = top3[0]['config']
        print(f"\nRetraining best model (val_acc={top3[0]['val_acc']:.4f})...")
        (X_train, y_train), (X_test, y_test) = load_data('mnist')
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
        y_train_onehot = one_hot_encode(y_train)
        y_val_onehot = one_hot_encode(y_val)

        best_args = type('Args', (), best_cfg)()
        model = NeuralNetwork(best_args)
        model.train(X_train, y_train_onehot, epochs=best_args.epochs, batch_size=best_args.batch_size, val_X=X_val, val_y=y_val_onehot)
        np.save('src/best_model.npy', model.get_weights())
        print("Saved best model to src/best_model.npy")

        if args.use_wandb:
            wandb.finish()
        print("Experiment 2.2 completed.")

    elif args.experiment == '2.7':
        sweep_project = args.sweep_name
        if not sweep_project:
            print("ERROR: --sweep_name is required for experiment 2.7")
            print("Usage: python3 src/experiment.py --experiment 2.7 --use_wandb --wandb_project <analysis_project> --sweep_name <sweep_project>")
            return

        if args.use_wandb:
            wandb.init(project=args.wandb_project, name="2.7_Overfitting_Analysis", group=args.experiment)

        api = wandb.Api()
        runs = list(api.runs(f'rishabh-mishra-cer16-iitmaana/{sweep_project}'))
        print(f"Total runs fetched from sweep '{sweep_project}': {len(runs)}")

        # Collect all run data with history
        all_runs_data = []
        for run in runs:
            try:
                history = run.history()
                train_acc = run.summary.get('final_train_acc', run.summary.get('train_acc', 0))
                test_acc = run.summary.get('final_test_acc', 0)
                gap = train_acc - test_acc
                all_runs_data.append({
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'gap': gap,
                    'config': run.config,
                    'run_name': run.name,
                    'history': history,
                })
            except Exception as e:
                print(f"Error processing run {run.name}: {e}")
                continue

        print(f"Total runs analyzed: {len(all_runs_data)}")

        if len(all_runs_data) == 0:
            print("ERROR: No runs found.")
            if args.use_wandb:
                wandb.finish()
            return

        # Sort by gap (train - test) descending → largest overfitting first
        all_runs_data.sort(key=lambda d: d['gap'], reverse=True)
        top3_overfit = all_runs_data[:3]

        # --- Overlay plot: train_acc, test_acc, val_loss for ALL runs ---
        colors_cycle = plt.cm.tab10(np.linspace(0, 1, min(len(all_runs_data), 10)))
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))

        for i, rd in enumerate(all_runs_data):
            history = rd['history']
            # Filter to epoch-level rows only (drop rows where val_acc is NaN)
            if 'val_acc' in history.columns:
                history = history.dropna(subset=['val_acc']).reset_index(drop=True)
            epochs = list(range(1, len(history) + 1))
            c = colors_cycle[i % len(colors_cycle)]
            label = rd['run_name'][:20]

            if 'train_acc' in history.columns:
                axes[0].plot(epochs, history['train_acc'], color=c, alpha=0.6, linewidth=1, label=label)
            if 'val_loss' in history.columns:
                axes[2].plot(epochs, history['val_loss'], color=c, alpha=0.6, linewidth=1, label=label)
            # test_acc as a flat line across epochs
            axes[1].plot(epochs, [rd['test_acc']] * len(epochs), color=c, alpha=0.6, linewidth=1, linestyle='--', label=label)

        axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Train Accuracy')
        axes[0].set_title('Train Acc (all runs)'); axes[0].grid(True, alpha=0.3)
        axes[1].set_xlabel('Run'); axes[1].set_ylabel('Test Accuracy')
        axes[1].set_title('Test Acc (all runs)'); axes[1].grid(True, alpha=0.3)
        axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Val Loss')
        axes[2].set_title('Val Loss (all runs)'); axes[2].grid(True, alpha=0.3)

        fig.suptitle(f'Overlay: All Sweep Runs from {sweep_project}', fontsize=12)
        plt.tight_layout()
        if args.use_wandb:
            wandb.log({"overlay_all_runs": wandb.Image(fig)})
        plt.close(fig)

        # --- Top 3 overfit runs: combined train_acc, val_acc, test_acc, val_loss in a single plot ---
        os.makedirs('src/plots', exist_ok=True)
        for rank, rd in enumerate(top3_overfit):
            history = rd['history']
            # Filter to epoch-level rows only (drop rows where val_acc is NaN)
            if 'val_acc' in history.columns:
                history = history.dropna(subset=['val_acc']).reset_index(drop=True)
            epochs = list(range(1, len(history) + 1))
            config = rd['config']
            config_parts = [
                f"opt={config.get('optimizer','?')}",
                f"lr={config.get('learning_rate','?')}",
                f"act={config.get('activation','?')}",
                f"layers={config.get('num_layers','?')}",
                f"hs={config.get('hidden_size','?')}",
            ]
            plot_title = f"Overfit #{rank+1} (gap={rd['gap']:.4f}) | " + " | ".join(config_parts)

            fig, ax1 = plt.subplots(figsize=(12, 6))

            # Left y-axis: accuracies
            if 'train_acc' in history.columns:
                ax1.plot(epochs, history['train_acc'], 'r-', linewidth=2, label='Train Acc')
            if 'val_acc' in history.columns:
                ax1.plot(epochs, history['val_acc'], 'm-', linewidth=2, label='Val Acc')
            ax1.axhline(y=rd['test_acc'], color='g', linestyle='--', linewidth=2, label=f'Test Acc: {rd["test_acc"]:.4f}')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Accuracy')
            ax1.grid(True, alpha=0.3)

            # Right y-axis: val loss
            ax2 = ax1.twinx()
            if 'val_loss' in history.columns:
                ax2.plot(epochs, history['val_loss'], 'b-', linewidth=2, label='Val Loss')
            ax2.set_ylabel('Validation Loss')

            # Combined legend
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

            fig.suptitle(plot_title, fontsize=9, y=1.02)
            plt.tight_layout()

            # Save overfit plot locally
            fname = (f"overfit_opt-{config.get('optimizer','x')}_lr-{config.get('learning_rate','x')}"
                     f"_act-{config.get('activation','x')}_layers-{config.get('num_layers','x')}"
                     f"_hs-{config.get('hidden_size','x')}_bs-{config.get('batch_size','x')}"
                     f"_loss-{config.get('loss','x')}_wd-{config.get('weight_decay','x')}"
                     f"_init-{config.get('weight_init','x')}.png")
            plt.savefig(os.path.join('src', 'plots', fname), bbox_inches='tight', dpi=150)

            if args.use_wandb:
                wandb.log({f"overfit_rank{rank+1}_{rd['run_name']}_curves": wandb.Image(fig)})
            plt.close(fig)

        # --- Save top 3 overfit configs for this sweep to overfit_configs.log ---
        with open('src/overfit_configs.log', 'w') as f:
            f.write(f"Sweep: {sweep_project}\n")
            f.write("=" * 80 + "\n")
            f.write("TOP 3 OVERFITTING RUNS (ranked by train_acc - test_acc gap)\n")
            f.write("=" * 80 + "\n\n")
            for i, d in enumerate(top3_overfit):
                f.write(f"Rank {i+1}: train_acc={d['train_acc']:.4f}, test_acc={d['test_acc']:.4f}, gap={d['gap']:.4f}\n")
                f.write(f"Run: {d['run_name']}\n")
                f.write(json.dumps(d['config'], indent=4) + "\n\n")

        print(f"\nTop 3 overfitting configs saved to src/overfit_configs.log")
        print("\nRank | Train Acc | Test Acc | Gap    | Key Config")
        print("-" * 80)
        for i, d in enumerate(top3_overfit):
            cfg = d['config']
            key_info = f"lr={cfg.get('learning_rate', 'N/A')}, opt={cfg.get('optimizer', 'N/A')}, act={cfg.get('activation', 'N/A')}"
            print(f"{i+1:4d} | {d['train_acc']:.4f}    | {d['test_acc']:.4f}   | {d['gap']:.4f} | {key_info}")

        # --- Update global overfit_config.json (best overfitting config across all sweeps) ---
        existing_overfit = None
        if os.path.exists('src/overfit_config.json'):
            try:
                with open('src/overfit_config.json', 'r') as f:
                    existing_overfit = json.load(f)
            except Exception:
                pass

        best_overfit = top3_overfit[0]
        update_global = False
        if existing_overfit is None:
            update_global = True
        elif best_overfit['gap'] > existing_overfit.get('gap', 0):
            update_global = True

        if update_global:
            overfit_cfg = best_overfit['config'].copy()
            overfit_cfg.pop('wandb_project', None)
            with open('src/overfit_config.json', 'w') as f:
                json.dump({
                    'gap': best_overfit['gap'],
                    'train_acc': best_overfit['train_acc'],
                    'test_acc': best_overfit['test_acc'],
                    'config': overfit_cfg
                }, f, indent=4)
            print(f"\nUpdated src/overfit_config.json (gap={best_overfit['gap']:.4f})")
        else:
            print(f"\nGlobal overfit_config.json unchanged (existing gap={existing_overfit.get('gap', 0):.4f} >= new {best_overfit['gap']:.4f})")

        if args.use_wandb:
            wandb.log({
                "overfitting_summary": wandb.Table(
                    data=[[i+1, d['train_acc'], d['test_acc'], d['gap'],
                           d['config'].get('learning_rate', 'N/A'),
                           d['config'].get('optimizer', 'N/A'),
                           d['config'].get('activation', 'N/A')]
                          for i, d in enumerate(top3_overfit)],
                    columns=["Rank", "Train Acc", "Test Acc", "Gap", "Learning Rate", "Optimizer", "Activation"]
                )
            })
            wandb.finish()
        print("\nExperiment 2.7 completed.")

if __name__ == "__main__":
    main()
