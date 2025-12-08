#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
Symptom Relationship Analysis: Testing Universal vs Context-Specific Patterns
Tests whether symptom relationships (correlations) are consistent across datasets
This addresses the fundamental flaw: K-matching is invalid, but symptom relationships are testable
"""


# In[2]:


import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


DATASETS = {
    'D1-Swiss': '/content/D1_Swiss_processed.csv',
    'D2-Cultural': '/content/D2_Cultural_processed.csv',
    'D3-Academic': '/content/D3_Academic_processed.csv',
    'D4-Tech': '/content/D4_Tech_processed.csv',
}

FEATURE_COLUMNS = ["Depression", "Anxiety", "Stress", "Burnout"]


# In[4]:


def compute_all_correlations(df, feature_columns):
    """
    Compute all pairwise correlations between features.
    Returns a dict like: {"Depression-Anxiety": 0.62, ...}
    """
    correlations = {}

    for i, feature1 in enumerate(feature_columns):
        for j, feature2 in enumerate(feature_columns[i+1:], start=i+1):

            pair_name = f"{feature1}-{feature2}"
            corr, _ = pearsonr(df[feature1], df[feature2])

            correlations[pair_name] = corr

    return correlations


# In[ ]:


def analyze_correlation_universality(comparison_df):
    """
    Classify symptom relationships as Universal vs Context-Specific.

    Universal: Strong correlation (|r| > 0.3) AND consistent across datasets (CV < 0.30)
    Contextual: Weak correlation (|r| ≤ 0.3) OR inconsistent across datasets (CV ≥ 0.30)

    Thresholds:
    - 0.3: Cohen's medium effect size (established in psychology)
    - CV < 0.30: Coefficient of variation < 30% = consistent
    """
    universal_correlations = []
    contextual_correlations = []

    for col in comparison_df.columns:
        vals = comparison_df[col].values
        vals = vals[~np.isnan(vals)]

        if len(vals) < 2:
            continue

        mean_correlation = np.mean(vals)
        std_correlation = np.std(vals)
        # Use abs() to handle negative correlations correctly
        cv_correlation = std_correlation / abs(mean_correlation) if mean_correlation != 0 else np.inf

        # Classify: Universal = strong and consistent
        if abs(mean_correlation) > 0.3 and cv_correlation < 0.30:
            universal_correlations.append({
                'pair': col,
                'mean_corr': mean_correlation,
                'cv_corr': cv_correlation,
                'std_corr': std_correlation,
                'values': vals.tolist()
            })
        else:
            contextual_correlations.append({
                'pair': col,
                'mean_corr': mean_correlation,
                'std_corr': std_correlation,
                'cv_corr': cv_correlation,
                'values': vals.tolist()
            })

    return universal_correlations, contextual_correlations


# In[ ]:


print("Loading datasets...")
all_data = {}
for dataset_name, file_path in DATASETS.items():
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"{dataset_name} not found at {file_path}")
        continue
    missing_features = [feature for feature in FEATURE_COLUMNS if feature not in df.columns]
    if missing_features:
        print(f"Skipping {dataset_name} - missing required features: {', '.join(missing_features)}")
        continue
    all_data[dataset_name] = df
    print(f"Loaded {dataset_name} with {len(df)} rows")

print("\nComputing correlations...")
results = {}

for dataset_name, df in all_data.items():
    correlations = compute_all_correlations(df, FEATURE_COLUMNS)
    results[dataset_name] = correlations

# Create comparison dataframe
comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df.round(3)

print("\nCorrelation Comparison Table:")
print(comparison_df)


# In[ ]:


def interpret_effect_size(correlation):
    """Interpret correlation using Cohen's (1988) benchmarks."""
    abs_corr = abs(correlation)
    if abs_corr < 0.1:
        return "negligible"
    elif abs_corr < 0.3:
        return "small"
    elif abs_corr < 0.5:
        return "medium"
    else:
        return "large"

# Analyze universality
universal_correlations, contextual_correlations = analyze_correlation_universality(comparison_df)

# Print results with effect size interpretation
print("="*80)
print("UNIVERSAL RELATIONSHIPS")
print("="*80)
print(f"Found {len(universal_correlations)} universal relationship(s):\n")

for item in universal_correlations:
    effect = interpret_effect_size(item['mean_corr'])
    print(f"  {item['pair']}:")
    print(f"    Mean correlation: {item['mean_corr']:.3f} ({effect} effect)")
    print(f"    CV: {item['cv_corr']:.3f} ({item['cv_corr']*100:.1f}%)")
    print(f"    Std: {item['std_corr']:.3f}")
    print(f"    Values across datasets: {[f'{v:.3f}' for v in item['values']]}")
    print()

print("="*80)
print("CONTEXTUAL RELATIONSHIPS")
print("="*80)
print(f"Found {len(contextual_correlations)} contextual relationship(s):\n")

for item in contextual_correlations:
    effect = interpret_effect_size(item['mean_corr'])
    reason = []
    if abs(item['mean_corr']) <= 0.3:
        reason.append("weak correlation")
    if item['cv_corr'] >= 0.30:
        reason.append("inconsistent")

    print(f"  {item['pair']}:")
    print(f"    Mean correlation: {item['mean_corr']:.3f} ({effect} effect)")
    print(f"    CV: {item['cv_corr']:.3f} ({item['cv_corr']*100:.1f}%)")
    print(f"    Std: {item['std_corr']:.3f}")
    print(f"    Values across datasets: {[f'{v:.3f}' for v in item['values']]}")
    print(f"    Reason: {' OR '.join(reason) if reason else 'moderate'}")
    print()

# Summary with hypothesis testing
print("="*80)
print("HYPOTHESIS TESTING SUMMARY")
print("="*80)
total_pairs = len(comparison_df.columns)
universal_pct = len(universal_correlations) / total_pairs * 100
contextual_pct = len(contextual_correlations) / total_pairs * 100

print(f"Total symptom pairs analyzed: {total_pairs}")
print(f"Universal relationships: {len(universal_correlations)}/{total_pairs} ({universal_pct:.1f}%)")
print(f"Contextual relationships: {len(contextual_correlations)}/{total_pairs} ({contextual_pct:.1f}%)")



# In[ ]:


# Minimal visualization: Correlation comparison
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(comparison_df.columns))
width = 0.2

for i, dataset_name in enumerate(comparison_df.index):
    ax.bar(x + i*width, comparison_df.loc[dataset_name], width, 
           label=dataset_name, alpha=0.8)

ax.set_xlabel('Symptom Pairs', fontsize=11)
ax.set_ylabel('Correlation (r)', fontsize=11)
ax.set_title('Symptom Relationships Across Datasets', fontsize=12, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(comparison_df.columns, rotation=45, ha='right')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:


# Viz - Summary statistics table
summary_data = []
for item in universal_correlations + contextual_correlations:
    effect = interpret_effect_size(item['mean_corr'])
    classification = 'Universal' if item in universal_correlations else 'Contextual'
    summary_data.append({
        'Symptom Pair': item['pair'],
        'Mean r': f"{item['mean_corr']:.3f}",
        'Effect Size': effect,
        'CV': f"{item['cv_corr']:.3f}",
        'Std': f"{item['std_corr']:.3f}",
        'Classification': classification
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values('Classification', ascending=False)

print("="*80)
print("SUMMARY STATISTICS TABLE")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80)


# In[ ]:


# EXPORT RESULTS FOR REPORT
results_summary = {
    'total_pairs': len(comparison_df.columns),
    'universal_count': len(universal_correlations),
    'contextual_count': len(contextual_correlations),
    'universal_pairs': [item['pair'] for item in universal_correlations],
    'contextual_pairs': [item['pair'] for item in contextual_correlations],
    'comparison_table': comparison_df.to_dict(),
    'universal_details': universal_correlations,
    'contextual_details': contextual_correlations
}

print("="*80)
print("RESULTS EXPORTED")
print("="*80)
print("Results stored in 'results_summary' dictionary")
print("Use this for your report/presentation")
print("="*80)


# 
# 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




