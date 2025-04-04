import pandas as pd
import matplotlib.pyplot as plt
import io

# CSV data as a string from the document

# Read the CSV data into a DataFrame using StringIO
df = pd.read_csv("obj_detect_results_ft.csv")
print(df)
# Unique modes
modes = df['Mode'].unique()

# Function to create bar plot
def create_bar_plot(y_column, ylabel, title):
    fig, ax = plt.subplots(figsize=(10, 6))
    bar_width = 0.15
    index = range(len(modes))
    
    # Plot bars for each rounding value
    for i, rounding in enumerate(range(1, 7)):
        values = [df[(df['Mode'] == mode) & (df['Rounding'] == rounding)][y_column].values[0] 
                  if not df[(df['Mode'] == mode) & (df['Rounding'] == rounding)].empty 
                  else 0 for mode in modes]
        plt.bar([x + bar_width * i for x in index], values, bar_width, label=f'Rounding {rounding}')
    
    plt.xlabel('Mode')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks([i + bar_width * 2.5 for i in index], modes)  # Center the x-ticks
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"bar_plot_{title}.pdf", bbox_inches='tight')
    plt.show()

# Create three separate bar plots
create_bar_plot('Mean Latency (ms)', 'Mean Latency (ms)', 'Mean Latency by Mode and Rounding')
create_bar_plot('Std Latency (ms)', 'Std Latency (ms)', 'Standard Deviation of Latency by Mode and Rounding')
create_bar_plot('mAP@0.5:0.95', 'mAP@0.5:0.95', 'mAP@0.5:0.95 by Mode and Rounding')