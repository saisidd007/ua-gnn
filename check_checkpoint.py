"""
Script to check the results from epoch 20 checkpoint
"""
import torch
import json
from datetime import datetime
import numpy as np
from src.models.enhanced_gnn import EnhancedTrafficGNN, create_enhanced_model
from src.utils.enhanced_dataset import create_enhanced_dataset

def load_and_check_checkpoint():
    # Load checkpoint
    print("Loading checkpoint from epoch 20...")
    checkpoint = torch.load('results/enhanced_checkpoint_epoch_20.pt')
    
    print("\n📊 Training Progress up to Epoch 20:")
    print("=" * 60)
    
    # Print training metrics
    history = checkpoint['history']
    print(f"\nTraining Metrics at Epoch 20:")
    print(f"Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Validation Loss: {history['val_loss'][-1]:.6f}")
    
    # Get the last validation metrics
    last_val_metrics = history['val_metrics'][-1]
    
    print(f"\nPerformance Metrics:")
    print(f"MAE: {last_val_metrics['mae']:.4f}")
    print(f"RMSE: {last_val_metrics['rmse']:.4f}")
    print(f"MAPE: {last_val_metrics['mape']:.2f}%")
    print(f"R² Score: {last_val_metrics['r2_score']:.4f}")
    
    if 'total_uncertainty' in last_val_metrics:
        print(f"\n📊 Uncertainty and Confidence Metrics:")
        print(f"Mean Aleatoric Uncertainty: {last_val_metrics['mean_aleatoric_uncertainty']:.4f}")
        print(f"Mean Epistemic Uncertainty: {last_val_metrics['mean_epistemic_uncertainty']:.4f}")
        print(f"Total Uncertainty: {last_val_metrics['total_uncertainty']:.4f}")
        conf_interval = 1.96 * np.sqrt(last_val_metrics['total_uncertainty'])
        print(f"95% Confidence Interval: ±{conf_interval:.4f}")
        if 'confidence_coverage' in last_val_metrics:
            coverage = last_val_metrics['confidence_coverage'] * 100
            print(f"Confidence Coverage: {coverage:.1f}% of true values within interval")
    
    print("\nLearning Progress:")
    print(f"Starting Learning Rate: {history['learning_rates'][0]:.2e}")
    print(f"Final Learning Rate: {history['learning_rates'][-1]:.2e}")
    
    # Plot if matplotlib is available
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(15, 10))
        
        # Plot training and validation loss
        plt.subplot(2, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot MAE
        plt.subplot(2, 2, 2)
        train_mae = [m['mae'] for m in history['train_metrics']]
        val_mae = [m['mae'] for m in history['val_metrics']]
        plt.plot(train_mae, label='Train MAE')
        plt.plot(val_mae, label='Val MAE')
        plt.title('MAE over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True)
        
        # Plot learning rate
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rates'])
        plt.title('Learning Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log')
        plt.grid(True)
        
        # Plot R² score
        plt.subplot(2, 2, 4)
        train_r2 = [m['r2_score'] for m in history['train_metrics']]
        val_r2 = [m['r2_score'] for m in history['val_metrics']]
        plt.plot(train_r2, label='Train R²')
        plt.plot(val_r2, label='Val R²')
        plt.title('R² Score over epochs')
        plt.xlabel('Epoch')
        plt.ylabel('R²')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('results/training_progress_epoch_20.png')
        print(f"\n✅ Training progress plot saved to results/training_progress_epoch_20.png")
        
    except ImportError:
        print("\nNote: matplotlib not available for plotting")

if __name__ == "__main__":
    load_and_check_checkpoint()