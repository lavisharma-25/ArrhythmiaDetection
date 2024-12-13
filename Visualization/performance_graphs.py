import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_performance(performance, output_dir):
    """
    Plot and save performance metrics for all models.
    """
    performance_df = pd.DataFrame(performance)
    performance_df.set_index('Model', inplace=True)

    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_df.plot(kind='barh', ax=ax, color=['#76D7C4', '#F5B7B1', '#F7DC6F', '#F39C12'], width=0.8)
    ax.set_xlabel('Score')
    ax.set_title('Model Performance Comparison')
    plt.tight_layout()

    # Save the figure
    image_path = os.path.join(output_dir, 'model_performance_comparison.png')
    plt.savefig(image_path, dpi=300)
    print(f"Performance comparison saved to: {image_path}")