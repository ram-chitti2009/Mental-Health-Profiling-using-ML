"""
Run H3 Validation from Saved h3_data.pkl
Tests if profile membership is associated with therapy utilization(chisq, cramers v, standardized residuals)
"""



from google.colab import drive
from sklearn.model_selection import train_test_split
drive.mount('/content/drive')

import os
import pickle
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

RANDOM_SEED = 42
H3_DATA_PATH = "/content/drive/MyDrive/CAPTURE/h3_data.pkl"
DATASET_PATH = "/content/D1_Swiss_processed.csv"


print("="*70)
print("H3 VALIDATION: Testing Clinical Utility of Profiles")
print("="*70)
print("Hypothesis H3: Profile membership is associated with therapy utilization")
print("Using FULL dataset (train+val+test) for maximum statistical analysis")
print("="*70)

print("Loading H3 data...")
with open(H3_DATA_PATH, 'rb') as f:
    h3_data = pickle.load(f)


print("H3 data loaded successfully")
print(f"  - Train+val clusters: {h3_data['cluster_labels_all'].shape}")
print(f"  - Test clusters: {h3_data['test_cluster_assignments'].shape}")
print(f"  - Centroids: {h3_data['cluster_centroids'].shape}")
print(f"  - Best k: {h3_data['best_k']}")

print("Loading dataset...")
df = pd.read_csv(DATASET_PATH)

if "PSYT_Therapy_Use" not in df.columns:
    raise ValueError("PSYT_Therapy_Use column not found in dataset")

y_therapy = df["PSYT_Therapy_Use"].values

#spli therapy data the same way we did the train+val+test split for the h3 data use the same seed

print("Splitting therapy data...")
train_val_therapy, test_therapy = train_test_split(
    y_therapy,
    test_size=0.2,
    random_state=RANDOM_SEED
)

y_therapy_aligned = np.concatenate([train_val_therapy, test_therapy])
all_cluster_labels = np.concatenate([
    h3_data['cluster_labels_all'],
    h3_data['test_cluster_assignments']
])

assert len(y_therapy_aligned) == len(all_cluster_labels), "Misalignment"
print(f"\n✓ Data aligned: {len(all_cluster_labels)} samples")
print(f"  Therapy use rate: {y_therapy_aligned.mean():.2%} ({y_therapy_aligned.sum()}/{len(y_therapy_aligned)})")

#create contingency table for chi2 test
print("\nChi-Square Test for Independence:")
print("="*70)
contingency = pd.crosstab(all_cluster_labels, y_therapy_aligned)
chi2, p, dof, expected = chi2_contingency(contingency)

print("   Contingency Table:")
print(contingency)
print(f"\n   Chi-square statistic: χ² = {chi2:.4f}")
print(f"   Degrees of freedom: df = {dof}")
print(f"   p-value: p = {p:.6f}")

alpha = 0.05


if p < alpha:
    print(f"\n   ✓ SIGNIFICANT (p < {alpha}): Profile membership IS associated with therapy utilization")
    print("   → H3 VALIDATED: Profiles have clinical utility")
else:
    print(f"\n   ✗ NOT SIGNIFICANT (p >= {alpha}): No association detected")
    print("   → H3 NOT VALIDATED")

print("\nCramer V Effect Size:")
print("="*70)
n = contingency.values.sum()
min_dim = min(contingency.shape)
cramers_v = np.sqrt(chi2 / (n * (min_dim - 1)))
print(f"   Cramer's V: {cramers_v:.4f}")


if cramers_v < 0.10:
    effect_size = "negligible"
elif cramers_v < 0.30:
    effect_size = "small"
elif cramers_v < 0.50:
    effect_size = "medium"
else:
    effect_size = "large"

print(f"   Effect size: {effect_size}")


#post-hoc analysis - standardized residuals to see which profiles deviate from expected

print("\n3. Post-Hoc Analysis: Standardized Residuals:")
print("Post-hoc analysis to identify profiles that deviate from expected")
print("-"*70)
print("   (Values > |2| indicate significant deviation from expected)")

residuals = (contingency.values - expected) / np.sqrt(expected + 1e-10)
residuals_df = pd.DataFrame(
    residuals,
    index=[f'P{k+1}' for k in range(h3_data['best_k'])],
    columns=['No Therapy', 'Therapy']
)
print("   Standardized Residuals:")
print(residuals_df.round(3))

#Identify profiles that deviate from expected
print("\n   Significant deviations:")

has_significant_deviation = False

for i in range(h3_data['best_k']):
    for j in range(2):
        if abs(residuals[i, j]) > 2:
            has_significant_deviation = True
            profile_name = f'P{i+1}'
            therapy_status = 'Therapy' if j == 1 else 'No Therapy'
            direction = 'Higher' if residuals[i, j] > 0 else 'Lower'
            print(f"      • {profile_name} - {therapy_status}: {direction} than expected (residual = {residuals[i, j]:.2f})")
if not has_significant_deviation:
    print("   No significant deviations found")

# Final Summary
print("\n" + "="*70)
print("H3 VALIDATION SUMMARY:")
print("="*70)
print(f"Dataset: {h3_data['dataset_name']} (N={len(all_cluster_labels)})")
print(f"Chi-square: X^2 = {chi2:.4f}, p = {p:.6f}, df = {dof}")
print(f"Cramér's V = {cramers_v:.4f} ({effect_size} effect)")
print(f"H3 Status: {' VALIDATED' if p < alpha else ' NOT VALIDATED'}")

if p < alpha:
    print("\nConclusion:")
    print("  Profiles demonstrate clinical utility by predicting therapy utilization.")
    print("  This supports the use of these profiles for targeted mental health interventions.")
else:
    print("\nConclusion:")
    print("  No significant association found between profile membership and therapy utilization.")
    print("  Profiles may not demonstrate clinical utility in this context.")

print("="*70)
