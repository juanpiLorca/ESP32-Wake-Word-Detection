import numpy as np
import matplotlib.pyplot as plt

def plot_training_history(history):
    metrics = history.history
    
    # Best epoch (min validation loss)
    epoch_stop_idx = np.argmin(metrics['val_loss'])
    epch_stop = history.epoch[epoch_stop_idx]

    plt.style.use('default')

    plt.figure(figsize=(10, 12))

    # ---------------------- LOSS (TOP) ----------------------
    plt.subplot(2, 1, 1)
    plt.plot(history.epoch, metrics['loss'],
             linestyle='-', marker='o', linewidth=2, markersize=5,
             label='Train Loss')

    plt.plot(history.epoch, metrics['val_loss'],
             linestyle='--', marker='x', linewidth=2, markersize=6,
             label='Val Loss')

    # Highlight best epoch
    plt.scatter(epch_stop, metrics['val_loss'][epoch_stop_idx],
                color='red', s=80, zorder=5,
                label=f'Best Epoch={epch_stop}')

    plt.xlabel("Epoch")
    plt.ylabel("Loss (CrossEntropy)")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(False)

    # ------------------ ACCURACY (BOTTOM) ------------------
    plt.subplot(2, 1, 2)
    plt.plot(history.epoch, 100*np.array(metrics['accuracy']),
             linestyle='-', marker='o', linewidth=2, markersize=5,
             label='Train Accuracy')

    plt.plot(history.epoch, 100*np.array(metrics['val_accuracy']),
             linestyle='--', marker='x', linewidth=2, markersize=6,
             label='Val Accuracy')

    plt.ylim([0, 100])
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy [%]")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(False)

    plt.tight_layout()
    plt.savefig("imgs/training_history.svg", format='svg')
    plt.show()
