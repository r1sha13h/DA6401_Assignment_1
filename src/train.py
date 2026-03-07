"""
Main Training Script
Entry point for training neural networks with command-line arguments
"""

import argparse
import json
import numpy as np
import wandb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sys
import os

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ann.neural_network import NeuralNetwork
from utils.data_loader import load_data, one_hot_encode, create_batches

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Train a neural network')

    # Dataset and basic training parameters
    parser.add_argument('-d', '--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion_mnist'],
                        help='Dataset to use: mnist or fashion_mnist')
    parser.add_argument('-e', '--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Mini-batch size')

    # Loss and optimizer
    parser.add_argument('-l', '--loss', type=str, default='cross_entropy',
                        choices=['cross_entropy', 'mean_squared_error', 'mse'],
                        help='Loss function')
    parser.add_argument('-o', '--optimizer', type=str, default='sgd',
                        choices=['sgd', 'momentum', 'nag', 'rmsprop'],
                        help='Optimizer')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0.0,
                        help='Weight decay (L2 regularization)')

    # Network architecture
    parser.add_argument('-nhl', '--num_layers', type=int, default=2,
                        help='Number of hidden layers')
    parser.add_argument('-sz', '--hidden_size', type=int, nargs='+', default=[128],
                        help='Hidden layer sizes (one per layer, or single value replicated for all layers)')
    parser.add_argument('-a', '--activation', type=str, default='relu',
                        choices=['relu', 'sigmoid', 'tanh'],
                        help='Activation function for hidden layers')
    parser.add_argument('-w_i', '--weight_init', type=str, default='xavier',
                        choices=['random', 'xavier', 'zeros'],
                        help='Weight initialization method')

    # Weights & Biases
    parser.add_argument('-w_p', '--wandb_project', type=str, default='mlp-numpy',
                        help='Weights & Biases project name')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')

    # Model saving
    parser.add_argument('--save_model', action='store_true',
                        help='Save the best model')
    parser.add_argument('--model_save_path', type=str, default='best_model.npy',
                        help='Path to save trained model')
    parser.add_argument('--config_save_path', type=str, default='best_config.json',
                        help='Path to save best configuration')

    args = parser.parse_args()

    # Ensure hidden_size matches num_layers
    if len(args.hidden_size) == 1:
        args.hidden_size = args.hidden_size * args.num_layers
    elif len(args.hidden_size) < args.num_layers:
        args.hidden_size.extend([args.hidden_size[-1]] * (args.num_layers - len(args.hidden_size)))
    elif len(args.hidden_size) > args.num_layers:
        args.hidden_size = args.hidden_size[:args.num_layers]

    return args

def update_best_configs_log(config, val_acc, log_path='best_configs.log'):
    """Update best_configs.log with top 5 configs by val_acc (descending order)."""
    entries = []
    
    # Read existing entries if file exists
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            # Parse existing entries
            import re
            pattern = r'Rank (\d+): val_acc ([\d.]+)\n(\{.*?\})'
            matches = re.findall(pattern, content, re.DOTALL)
            for rank, acc, cfg_str in matches:
                try:
                    cfg = json.loads(cfg_str)
                    entries.append({'val_acc': float(acc), 'config': cfg})
                except:
                    pass
    
    # Add current entry
    entries.append({'val_acc': val_acc, 'config': config})
    
    # Sort by val_acc descending and keep top 5
    entries = sorted(entries, key=lambda x: x['val_acc'], reverse=True)[:5]
    
    # Write back
    with open(log_path, 'w') as f:
        for i, entry in enumerate(entries):
            f.write(f"Rank {i+1}: val_acc {entry['val_acc']:.4f}\n")
            f.write(json.dumps(entry['config'], indent=4) + "\n\n")
    
    print(f"Updated {log_path} with {len(entries)} entries (best val_acc: {entries[0]['val_acc']:.4f})")


def update_overfit_configs_log(config, train_acc, test_acc, log_path='overfit_configs.log'):
    """Update overfit_configs.log with configs where train_acc >= 0.9, sorted by gap (train-test) descending."""
    if train_acc < 0.9:
        return  # Skip if train_acc < 0.9
    
    gap = train_acc - test_acc
    entries = []
    
    # Read existing entries if file exists
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            # Parse existing entries
            import re
            pattern = r'Rank (\d+): train-test gap ([\d.]+) \(train=([\d.]+), test=([\d.]+)\)\n(\{.*?\})'
            matches = re.findall(pattern, content, re.DOTALL)
            for rank, gap_val, tr_acc, te_acc, cfg_str in matches:
                try:
                    cfg = json.loads(cfg_str)
                    entries.append({
                        'gap': float(gap_val),
                        'train_acc': float(tr_acc),
                        'test_acc': float(te_acc),
                        'config': cfg
                    })
                except:
                    pass
    
    # Add current entry
    entries.append({
        'gap': gap,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'config': config
    })
    
    # Sort by gap descending
    entries = sorted(entries, key=lambda x: x['gap'], reverse=True)
    
    # Write back
    with open(log_path, 'w') as f:
        for i, entry in enumerate(entries):
            f.write(f"Rank {i+1}: train-test gap {entry['gap']:.4f} (train={entry['train_acc']:.4f}, test={entry['test_acc']:.4f})\n")
            f.write(json.dumps(entry['config'], indent=4) + "\n\n")
    
    print(f"Updated {log_path} with {len(entries)} entries (largest gap: {entries[0]['gap']:.4f})")


def main():
    """
    Main training function.
    """
    args = parse_arguments()

    # Load data
    print(f"Loading {args.dataset} dataset...")
    (X_train, y_train), (X_test, y_test) = load_data(args.dataset)

    # Split train into train and val
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    print(f"Train samples: {X_train.shape[0]}")
    print(f"Val samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    # One-hot encode labels
    y_train_onehot = one_hot_encode(y_train)
    y_val_onehot = one_hot_encode(y_val)
    y_test_onehot = one_hot_encode(y_test)

    # Initialize model
    print("\nInitializing model...")
    print(f"Architecture: {[784] + args.hidden_size + [10]}")
    print(f"Activation: {args.activation}")
    model = NeuralNetwork(args)

    # Initialize wandb if enabled
    if args.use_wandb:
        # Create run name from hyperparameter config
        run_name = f"opt={args.optimizer}_lr={args.learning_rate}_act={args.activation}_layers={args.num_layers}_hs={args.hidden_size[0] if args.hidden_size else 'N/A'}_bs={args.batch_size}_loss={args.loss}_wd={args.weight_decay}_init={args.weight_init}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            group="sweep_runs",
            config={
                "dataset": args.dataset,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "loss": args.loss,
                "optimizer": args.optimizer,
                "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay,
                "num_layers": args.num_layers,
                "hidden_size": args.hidden_size,
                "activation": args.activation,
                "weight_init": args.weight_init,
            }
        )

    # Training loop
    best_val_f1 = 0.0
    # Track metrics for plotting
    metrics_history = {'val_loss': [], 'val_acc': [], 'train_acc': []}
    
    for epoch in range(args.epochs):
        # Train one epoch
        model.train(X_train, y_train_onehot, epochs=1, batch_size=args.batch_size)

        # Validate
        val_logits = model.forward(X_val)
        val_loss = model.loss_fn(y_val_onehot, val_logits)
        val_pred = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_pred == y_val)
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        val_precision = precision_score(y_val, val_pred, average='weighted', zero_division=0)
        val_recall = recall_score(y_val, val_pred, average='weighted', zero_division=0)

        # Compute train accuracy for this epoch (on subset for speed)
        train_subset = min(1000, X_train.shape[0])
        train_pred = model.predict(X_train[:train_subset])
        train_acc = np.mean(train_pred == y_train[:train_subset])
        
        # Store metrics
        metrics_history['val_loss'].append(val_loss)
        metrics_history['val_acc'].append(val_acc)
        metrics_history['train_acc'].append(train_acc)

        # Log
        print(f"Epoch {epoch+1}/{args.epochs}: val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}")

        if args.use_wandb:
            wandb.log({
                'epoch': epoch,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'val_f1': val_f1,
                'val_precision': val_precision,
                'val_recall': val_recall,
            })

            # Log gradient norms
            grad_norms = [np.linalg.norm(layer.grad_W) for layer in model.layers]
            wandb.log({
                f'grad_norm_layer_{i}': norm 
                for i, norm in enumerate(grad_norms)
            })

            # Log weight statistics
            for i, layer in enumerate(model.layers):
                wandb.log({
                    f'weight_mean_layer_{i}': layer.W.mean(),
                    f'weight_std_layer_{i}': layer.W.std()
                })

            # Log sample predictions every 5 epochs
            if epoch % 5 == 0:
                sample_images = X_val[:10]
                sample_labels = y_val[:10]
                predictions = model.predict(sample_images)
                wandb.log({
                    "sample_predictions": wandb.Table(
                        data=[[img.tolist(), true, pred] for img, true, pred in 
                              zip(sample_images, sample_labels, predictions)],
                        columns=["image", "true_label", "predicted_label"]
                    )
                })

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            if args.save_model:
                np.save(args.model_save_path, model.get_weights())
                print(f"New best model saved with val_f1={best_val_f1:.4f}")

    # Create plot with config as title after training completes
    if args.use_wandb and metrics_history['val_loss']:
        import matplotlib.pyplot as plt
        
        epochs = list(range(1, len(metrics_history['val_loss']) + 1))
        
        # Create config string for title
        config_parts = [
            f"ds={args.dataset}",
            f"ep={args.epochs}",
            f"bs={args.batch_size}",
            f"lr={args.learning_rate}",
            f"opt={args.optimizer}",
            f"act={args.activation}",
            f"layers={args.num_layers}",
            f"hidden={args.hidden_size[0] if args.hidden_size else 'N/A'}",
            f"loss={args.loss}",
            f"init={args.weight_init}",
            f"wd={args.weight_decay}"
        ]
        plot_title = " | ".join(config_parts)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Plot 1: Val Loss
        axes[0].plot(epochs, metrics_history['val_loss'], 'b-o', linewidth=2, markersize=4)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Validation Loss')
        axes[0].set_title('Validation Loss vs Epoch')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Val Acc
        axes[1].plot(epochs, metrics_history['val_acc'], 'g-o', linewidth=2, markersize=4)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Validation Accuracy')
        axes[1].set_title('Validation Accuracy vs Epoch')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Train Acc
        axes[2].plot(epochs, metrics_history['train_acc'], 'r-o', linewidth=2, markersize=4)
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Training Accuracy')
        axes[2].set_title('Training Accuracy vs Epoch')
        axes[2].grid(True, alpha=0.3)
        
        fig.suptitle(plot_title, fontsize=10, y=1.02)
        plt.tight_layout()
        
        # Log to wandb
        wandb.log({"training_curves": wandb.Image(fig)})
        plt.close(fig)
        print(f"Logged training curves plot with config: {plot_title[:80]}...")

    # Log final accuracies
    final_train_acc = np.mean(model.predict(X_train[:2000]) == y_train[:2000])
    final_test_acc = np.mean(model.predict(X_test[:2000]) == y_test[:2000])
    final_val_acc = np.mean(model.predict(X_val) == y_val)
    if args.use_wandb:
        wandb.log({"final_train_acc": final_train_acc, "final_test_acc": final_test_acc})

    # Update best_configs.log with top 5 configs by val_acc
    config_dict = {
        "loss": args.loss,
        "epochs": args.epochs,
        "dataset": args.dataset,
        "optimizer": args.optimizer,
        "activation": args.activation,
        "batch_size": args.batch_size,
        "num_layers": args.num_layers,
        "hidden_size": args.hidden_size,
        "weight_init": args.weight_init,
        "weight_decay": args.weight_decay,
        "learning_rate": args.learning_rate
    }
    script_dir = os.path.dirname(os.path.abspath(__file__))
    update_best_configs_log(config_dict, final_val_acc, log_path=os.path.join(script_dir, 'best_configs.log'))
    
    # Update overfit_configs.log (only if train_acc >= 0.9, in src folder)
    update_overfit_configs_log(config_dict, final_train_acc, final_test_acc, log_path=os.path.join(script_dir, 'overfit_configs.log'))

    # Save config
    if args.save_model:
        with open(args.config_save_path, 'w') as f:
            json.dump(vars(args), f)

        print(f"Best model saved to {args.model_save_path}")
        print(f"Config saved to {args.config_save_path}")

    if args.use_wandb:
        wandb.finish()

    print("Training complete!")


if __name__ == '__main__':
    main()
