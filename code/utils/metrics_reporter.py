import matplotlib.pyplot as plt
import os
import torch 

CODEOCEAN_BASE_DIR = os.environ.get('CODEOCEAN_BASE_DIR', '/')

RESULTS_PATH = os.path.join(CODEOCEAN_BASE_DIR, 'results')
MODEL_SAVE_PATH = os.path.join(RESULTS_PATH, 'iris_nn_model.pth')
REPORT_SAVE_PATH = os.path.join(RESULTS_PATH, 'analysis_report.txt')
TRAINING_PLOT_PATH = os.path.join(RESULTS_PATH, 'training_metrics.png')
NUM_EPOCHS = 50 
BATCH_SIZE = 16 
LEARNING_RATE = 0.01


def plot_metrics(train_losses, test_accuracies):
    epochs = range(1, NUM_EPOCHS + 1)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'o-', label='Training Loss')
    plt.title('Training Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, test_accuracies, 'o-', label='Test Accuracy', color='orange')
    plt.title('Test Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(TRAINING_PLOT_PATH)
    print(f"Training metrics plot saved to {TRAINING_PLOT_PATH}")


def generate_report(train_losses, test_accuracies, classes, final_model_path):
    with open(REPORT_SAVE_PATH, 'w') as f:
        f.write("--- Neural Network Analysis Report (Tabular Data) ---\n\n")
        f.write(f"Date: {torch.__version__}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Dataset: Iris (Tabular)\n")
        f.write(f"Number of Epochs: {NUM_EPOCHS}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Learning Rate: {LEARNING_RATE}\n\n")

        f.write("Training Summary:\n")
        for i in range(NUM_EPOCHS):
            f.write(f"  Epoch {i+1}: Loss = {train_losses[i]:.3f}, Test Accuracy = {test_accuracies[i]:.2f}%\n")
        f.write(f"\nFinal Test Accuracy: {test_accuracies[-1]:.2f}%\n")
        f.write(f"Model saved to: {final_model_path}\n")
        f.write(f"Training plots saved to: {TRAINING_PLOT_PATH}\n")
        f.write(f"Classes: {classes}\n")

    print(f"Analysis report saved to {REPORT_SAVE_PATH}")
