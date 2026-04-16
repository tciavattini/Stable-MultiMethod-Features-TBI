import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

df = pd.read_pickle('../results/patient_classification/df_binary_percentile.pkl')

# Compute baseline composite severity
# Directionally align so higher = worse for all domains
# Fatigue and pain: already higher = worse
# Mood and overall symptoms: higher = better, so invert
baseline_severity = df[['T0_severe_fatigue_rate', 'T0_muscle_pain_rate']].mean(axis=1) \
                  - df[['T0_mood_rate', 'T0_symp_today_rate']].mean(axis=1)

# Composite improvement score (already computed as 'composite_score' or similar)
# Adjust the column name to match your dataset
improvement = df['composite_z']  # adjust column name

# Drop NaN
mask = baseline_severity.notna() & improvement.notna()
x = baseline_severity[mask]
y = improvement[mask]

r, p = pearsonr(x, y)

fig, ax = plt.subplots(figsize=(6, 5))
colors = df.loc[mask, 'y'].map({1: '#2E75B6', 0: '#C0392B'})
ax.scatter(x, y, c=colors, alpha=0.6, edgecolors='white', linewidth=0.5)
ax.set_xlabel('Baseline composite severity (higher = worse somatic burden)', fontsize=11)
ax.set_ylabel('Composite improvement score', fontsize=11)
ax.set_title(f'Pearson r = {r:.3f}, p = {p:.4f}', fontsize=11)
ax.axhline(0, color='grey', linestyle='--', linewidth=0.5)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#2E75B6', markersize=8, label='High responders'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#C0392B', markersize=8, label='Non-responders'),
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.savefig('figures/suppfig_regression_to_mean.pdf', dpi=300)
plt.show()

print(f"Pearson r = {r:.3f}, p = {p:.4f}")