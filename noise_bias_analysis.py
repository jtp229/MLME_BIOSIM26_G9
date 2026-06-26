import pandas as pd
import numpy as np

# 1. Load the collected data
raw_data_file = "noise_bias_data.csv"
try:
    df = pd.read_csv(raw_data_file)
except FileNotFoundError:
    print(f"Error: Could not find '{raw_data_file}'. Please run your probe script first.")
    exit(1)

print("=== MULTI-SCALE NOISE AND BIAS ANALYSIS ===\n")

# 2. Group by recipe and scale to compute mean and sample standard deviation (noise)
summary = df.groupby(['recipe_name', 'scale'])['Y'].agg(['mean', 'std', 'count']).reset_index()

# Separate pilot data to act as the reference truth for bias calculation
pilot_df = summary[summary['scale'] == 'pilot'][['recipe_name', 'mean']].rename(columns={'mean': 'pilot_mean'})

# Merge back to calculate bias: (scale_mean - pilot_mean)
analytics = pd.merge(summary, pilot_df, on='recipe_name')
analytics['bias'] = analytics['mean'] - analytics['pilot_mean']

# Clean up presentation names
analytics = analytics.rename(columns={'mean': 'Mean_Y', 'std': 'Noise_StdDev', 'count': 'Reps'})
analytics = analytics[['recipe_name', 'scale', 'Reps', 'Mean_Y', 'Noise_StdDev', 'bias']]

# Print the granular table to terminal
print(analytics.to_string(index=False))

# --- Save granular analysis to CSV ---
granular_csv = "scale_recipe_analytics.csv"
analytics.to_csv(granular_csv, index=False)
print(f"\n-> Saved granular analytics to: {granular_csv}")


print("\n=== AGGREGATED METRICS PER SCALE ===")
aggregated_rows = []

for scale in ['micro', 'bench', 'pilot']:
    scale_data = analytics[analytics['scale'] == scale]
    avg_noise = scale_data['Noise_StdDev'].mean()
    avg_bias = scale_data['bias'].mean()
    
    # Terminal display
    print(f"Scale: {scale.upper()}")
    print(f"  Avg Noise (StdDev): {avg_noise:.5f}" if not np.isnan(avg_noise) else "  Avg Noise (StdDev): NaN (Insufficient Reps)")
    print(f"  Avg Bias vs Pilot : {avg_bias:.5f}")
    
    # Store for CSV export
    aggregated_rows.append({
        "scale": scale,
        "avg_noise_stddev": avg_noise if not np.isnan(avg_noise) else "",
        "avg_bias_vs_pilot": avg_bias
    })

# --- Save aggregated metrics to CSV ---
aggregated_df = pd.DataFrame(aggregated_rows)
aggregated_csv = "scale_aggregated_metrics.csv"
aggregated_df.to_csv(aggregated_csv, index=False)
print(f"-> Saved aggregated metrics to: {aggregated_csv}")

"""
================================================================================
STRATEGIC DISCUSSION & BO LOOP IMPLICATIONS (Task 2)
================================================================================
Based on the empirical evaluation of the probe data, the reactor scales exhibit
the following structural properties relative to the unbiased Pilot scale[cite: 15, 18]:

1. Microplate: Severe negative bias (~ -1.74 g/L), low absolute noise (sigma ~ 0.018)[cite: 13, 18].
2. Benchtop:   High positive bias (~ +2.57 g/L), intermediate noise (sigma ~ 0.074)[cite: 14, 18].
3. Pilot:      Ground truth reference (unbiased by definition), noise N/A[cite: 15, 18].

ALGORITHMIC IMPACT ON BAYESIAN OPTIMIZATION:
--------------------------------------------------------------------------------
- Naive Data Pooling is Defunct: Mixing raw (r, Y) points across scales in a standard 
  single-task GP will cause the optimizer to collapse toward Benchtop recipes, 
  falsely believing they yield the highest Pilot outputs.
  
- Multi-Fidelity Surrogate Modeling: We must implement an explicit multi-fidelity 
  architecture (e.g., Co-Kriging or a Linear Model of Coregionalization)[cite: 33, 57]. The 
  model must actively learn a scale-dependent scaling factor (rho) and a bias offset 
  function delta(r) to project low-fidelity trends onto the true Pilot space[cite: 31].

- Heteroskedastic Likelihood Noise: Rather than using a single global noise parameter, 
  the Gaussian Process likelihood must fix or initialize its scale-specific observation 
  noise parameters using the quantified standard deviations computed above.

- Cost-Aware Exploration: With an immense cost asymmetry (10 EUR Micro vs 2000 EUR Pilot)[cite: 13, 15, 16], 
  the acquisition function must optimize for 'Information Gain per Euro'[cite: 32, 33]. The loop 
  should leverage the ultra-cheap Micro scale to screen the global recipe topology, 
  using the Pilot scale sparingly to exploit and verify global optima[cite: 8, 10].
================================================================================
"""