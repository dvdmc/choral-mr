import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Set the font to Roboto
plt.rcParams['font.sans-serif'] = ['Roboto']
plt.rcParams['font.family'] = 'sans-serif'

# Read the CSV file into a DataFrame
df = pd.read_csv('vrp_cost_solution_time.csv')

min_costs = df.groupby('map')['total_cost'].min()

df['relative_gap'] = df.apply(lambda row: (row['total_cost'] - min_costs[row['map']]) / min_costs[row['map']], axis=1)
df['series'] = df.apply(lambda row: f'{row["map"]} ({row["num_tasks"]})', axis=1)

sns.set_theme(style="whitegrid")
plt.figure(figsize=(6, 2.5))
sns.lineplot(data=df, x='vrp_time', y='relative_gap', hue='series', marker='o')

plt.xlabel('VRP Solver Time (s)')
plt.ylabel('Rel. Gap (%)')

plt.legend(title='Map (#Tasks)', loc='upper right')
plt.tight_layout()

plt.savefig('relative_gap_vs_time.png', dpi=1000)

print("Plot saved as 'relative_gap_vs_time.png'")