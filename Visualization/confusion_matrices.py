import os
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def plot_confusion_matrices(cm_list, models, output_dir):
    """
    Plot and save confusion matrices for all models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()

    for idx, (model_name, cm) in enumerate(cm_list):
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=models[model_name].classes_)
        disp.plot(cmap='Blues', ax=axes[idx])
        axes[idx].set_title(f'Confusion Matrix: {model_name}')
        axes[idx].set_xlabel('Predicted', fontsize=12)
        axes[idx].set_ylabel('Actual', fontsize=12)

    # Adjust layout and save the figure
    plt.tight_layout()
    image_path = os.path.join(output_dir, 'confusion_matrices.png')
    plt.savefig(image_path, dpi=300)
    print(f"Confusion matrices saved to: {image_path}")