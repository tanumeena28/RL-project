import matplotlib.pyplot as plt
import seaborn as sns
import os

def generate_comparison_plot():
    # Data extracted from Section 7.1 of the research paper
    models = ['Prompt Engineering\n(Baseline)', 'Behavioral Cloning\n(Supervised)', 'CQL RL\n(Original Data)', 'CQL+ RL\n(Augmented Data)']
    success_rates = [36.00, 36.23, 48.67, 60.33]

    # Assign colors to highlight the "Before" vs "After"
    colors = ['#FF9999', '#FFCC99', '#99CCFF', '#0066CC']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, success_rates, color=colors)

    # Add data labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontweight='bold', fontsize=12)

    # Styling the plot
    plt.title('Student Problem-Solving Success Rate: Before vs After Offline RL', fontsize=16, fontweight='bold', pad=20)
    plt.ylabel('Success Rate (%)', fontsize=14)
    plt.ylim(0, 70)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Hide top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    # Save the plot
    save_path = os.path.join(os.path.dirname(__file__), 'metrics_accuracy.png')
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved successfully to {save_path}")

if __name__ == "__main__":
    generate_comparison_plot()
