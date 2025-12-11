import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    metrics = history.history
    # Find best epoch
    epoch_stop_idx = np.argmin(metrics['val_loss'])
    epch_stop = history.epoch[epoch_stop_idx]

    plt.style.use('default')  #

    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(history.epoch, 
            metrics['loss'], 
            linestyle='-', marker='o', linewidth=2, markersize=5,
            label='Train Loss')

    plt.plot(history.epoch, 
            metrics['val_loss'], 
            linestyle='--', marker='x', linewidth=2, markersize=6,
            label='Val Loss')

    # highlight best epoch
    plt.scatter(epch_stop, metrics['val_loss'][epoch_stop_idx],
                color='red', s=80, zorder=5, label='Best Epoch')

    plt.ylim([0, max(metrics['loss'] + metrics['val_loss'])])
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (CrossEntropy)", fontsize=12)
    plt.title("Training vs Validation Loss", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)

    plt.subplot(1, 2, 2)
    plt.plot(history.epoch,
            100*np.array(metrics['accuracy']), 
            linestyle='-', marker='o', linewidth=2, markersize=5,
            label='Train Accuracy')
    plt.plot(history.epoch,
            100*np.array(metrics['val_accuracy']), 
            linestyle='--', marker='x', linewidth=2, markersize=6,
            label='Val Accuracy')
    plt.ylim([0, 100])
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy [%]", fontsize=12)
    plt.title("Training vs Validation Accuracy", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(False)

    plt.tight_layout()
    plt.show()