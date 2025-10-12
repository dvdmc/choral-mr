import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the font to Roboto
plt.rcParams['font.sans-serif'] = ['Roboto']
plt.rcParams['font.family'] = 'sans-serif'

# Read the CSV file into a DataFrame
df = pd.read_csv('vrp_cost_solution_time.csv')

# Find the converged solution (minimum cost) for each map
# This creates a Series with 'map' as the index and the minimum 'total_cost' as the value.
min_costs = df.groupby('map')['total_cost'].min()

# Calculate the 'relative_gap' for each data point
df['relative_gap'] = df.apply(lambda row: (row['total_cost'] - min_costs[row['map']]) / min_costs[row['map']], axis=1)

# Create the new `series` column for the plot legend
df['series'] = df.apply(lambda row: f'{row["map"]} (V={row["num_tasks"]})', axis=1)

# Set the plot style
sns.set_theme(style="whitegrid")

# Create the line plot
plt.figure(figsize=(10, 4))
sns.lineplot(data=df, x='vrp_time', y='relative_gap', hue='series', marker='o')

# Set labels and title
plt.xlabel('VRP Solver Time (s)')
plt.ylabel('Rel. Gap to Converged Solution (%)')
# plt.title('Relative Gap to Converged Solution vs. VRP Solver Time')

# Adjust the legend position
plt.legend(title='Map (V=#Tasks)', loc='upper right')
plt.tight_layout()

# Save the plot
plt.savefig('relative_gap_vs_time.png', dpi=1000)

print("Plot saved as 'relative_gap_vs_time.png'")