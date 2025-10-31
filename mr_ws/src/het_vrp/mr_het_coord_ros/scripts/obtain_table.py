import pandas as pd
import os

csv_path = "results/results.csv"
latex_out = "tables/vrp_results.tex"

df = pd.read_csv(csv_path)

def agent_type_name(t):
    t = str(t).upper()
    if t in ["GROUND", "G"]:
        return "G"
    elif t in ["AERIAL", "A"]:
        return "A"
    else:
        return t

df["agent_type_norm"] = df["agent_type"].apply(agent_type_name)

# Create vehicle configuration string per experiment and method ===
vehicle_config_df = (
    df.groupby(["exp_id", "method"])
      .apply(lambda g: ",".join(g.sort_values("agent_id")["agent_type_norm"]))
      .reset_index(name="vehicle_config")
)
df = df.merge(vehicle_config_df, on=["exp_id", "method"], how="left")

agg_cols = ["total_cost", "distance", "time", "traversability", "collision", "safety"]

# Aggregate mean, std, min, max
summary = (
    df.groupby(["map", "method", "vehicle_config"])[agg_cols]
      .agg(["mean", "std", "min", "max"])
      .reset_index()
)

# Flatten MultiIndex columns
summary.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in summary.columns.values]

# Round numeric columns
numeric_cols = [c for c in summary.columns if any(m in c for m in ["mean", "std", "min", "max"])]
summary[numeric_cols] = summary[numeric_cols].round(3)

# Create mean ± std wrapped in $$ for LaTeX
for m in agg_cols:
    summary[f"{m}_latex"] = summary.apply(
        lambda row: f"${row[f'{m}_mean']} \\pm {row[f'{m}_std']}$", axis=1
    )

# Keep only identifiers + latex-formatted metrics
latex_cols = ["map", "method", "vehicle_config"] + [f"{m}_latex" for m in agg_cols]
latex_df = summary[latex_cols]

# Rename columns for readability
latex_df = latex_df.rename(columns={
    "map": "Map",
    "method": "Method",
    "vehicle_config": "Robots",
    "total_cost_latex": "Cost",
    "distance_latex": "Dist.",
    "time_latex": "Time",
    "traversability_latex": "Trav.",
    "collision_latex": "Coll.",
    "safety_latex": "Safety"
})

os.makedirs(os.path.dirname(latex_out), exist_ok=True)
latex_table = latex_df.to_latex(
    index=False,
    caption="Aggregated VRP performance metrics ($\\mathrm{mean \\pm std}$) per method, map, and vehicle configuration.",
    label="tab:vrp_results",
    escape=False,
    column_format="lllrrrrr"
)
with open(latex_out, "w") as f:
    f.write(latex_table)

print(f"LaTeX table saved to {latex_out}")
print(latex_df.head())
