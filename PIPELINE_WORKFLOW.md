# Autoencoder Pipeline Workflow Documentation

## Overview
This document describes the complete workflow for running the autoencoder-based mental health profiling pipeline on D1_Swiss_processed dataset with H3 validation.

## Quick Start

```bash
python d3_plswork.py
```

**Expected Runtime:** ~1.5-2 hours on A100 GPU

---

## Pipeline Stages

### Stage 1: Architecture Parameter Tuning (10-Fold CV)

**Purpose:** Find optimal neural network architecture

**Parameters Tested:**
- `hidden_sizes`: [4, 5, 6, 7, 8]
- `latent_dims`: [2, 3, 4]
- `activations`: [ReLU, Tanh, Sigmoid]
- `optimizers`: [Adam, SGD]
- `epochs`: [50, 75, 100]

**Total Experiments:** 5 × 3 × 3 × 2 × 3 × 10 = **2,700 experiments**

**Process:**
1. For each configuration, train autoencoder on train fold
2. Extract latent vectors from validation fold
3. Test K-means clustering with K=[2,3,4,5,6]
4. Evaluate using 4 metrics:
   - Silhouette Score (primary)
   - Calinski-Harabasz Index
   - Davies-Bouldin Index
   - Elbow Method (WCSS)
5. Select optimal K using consensus voting (2+ methods must agree)

**Selection Method:** Consensus voting across metrics, fallback to Silhouette

**Output:** Best architecture configuration (hidden_size, latent_dim, activation, optimizer, epochs, optimal_k)

**Time:** ~1-1.5 hours on A100

---

### Stage 2: Learning Parameter Optimization (10-Fold CV)

**Purpose:** Fine-tune training hyperparameters using best architecture from Stage 1

**Process:**
1. **LR Range Test:** Find optimal learning rate (1e-7 to 10)
2. **Grid Search:**
   - `batch_sizes`: [32, 64, 128]
   - `weight_decay`: [0, 1e-4, 1e-3]
   - `momentum`: [0.5, 0.9, 0.95] (if SGD) or N/A (if Adam)

**Total Experiments:** ~90-270 (depends on optimizer)

**Selection Method:** Consensus voting on clustering metrics

**Output:** Best learning parameters (learning_rate, batch_size, weight_decay, momentum)

**Time:** ~10-15 minutes on A100

---

### Stage 3: Final Model Training

**Purpose:** Train final model on all train+val data with best hyperparameters

**Process:**
1. Train autoencoder on 80% of data (train+val combined)
2. Extract latent vectors
3. Perform K-means clustering with optimal K
4. Compute cluster centroids
5. Extract profile characteristics (mean symptom levels per cluster)

**Output:**
- Trained model state
- Latent vectors
- Cluster assignments
- Cluster centroids
- Profile summary (mean Depression, Anxiety, Stress, Burnout per profile)

**Time:** ~5 minutes

---

### Stage 4: Test Set Evaluation

**Purpose:** Evaluate model generalization on held-out test set (20%)

**Process:**
1. Encode test data using trained autoencoder
2. Assign test samples to clusters using trained centroids (no retraining)
3. Compute reconstruction loss
4. Evaluate clustering quality metrics

**Output:**
- Test reconstruction loss
- Test clustering metrics (Silhouette, CH, DB)
- Generalization assessment (overfitting check)

**Time:** ~1 minute

---

### Stage 5: H3 Validation (Clinical Utility Testing)

**Purpose:** Test if profile membership predicts therapy utilization

**Prerequisites:**
- Dataset must be Swiss (D1_Swiss_processed)
- Must have `PSYT_Therapy_Use` column
- Early check performed before training starts

**Process:**
1. Combine train+val+test cluster assignments
2. Align with therapy utilization labels
3. Create contingency table (Profiles × Therapy Use)
4. Run Chi-square test of independence
5. Compute Cramér's V (effect size)
6. Post-hoc analysis: Standardized residuals

**Statistical Tests:**
- **Chi-square (χ²):** Tests independence (H0: no association)
- **P-value:** Probability of observing this data if H0 is true
- **Cramér's V:** Effect size (0.1=small, 0.3=medium, 0.5=large)
- **Standardized Residuals:** Which profiles deviate from expected

**Interpretation:**
- **p < 0.05:** Significant association → H3 VALIDATED
- **p ≥ 0.05:** No significant association → H3 NOT VALIDATED
- **Cramér's V:** Strength of association
- **Residuals > |2|:** Significant deviation (over/under-representation)

**Output:**
- Contingency table
- Chi-square statistic, p-value, degrees of freedom
- Cramér's V and effect size interpretation
- Standardized residuals table
- H3 validation conclusion

**Time:** ~1 minute

---

## Saved Outputs

All results are saved to: `D1_Swiss_processed_results/`

### 1. Model File
- **File:** `D1_Swiss_processed_model.pth`
- **Contents:**
  - Model state dictionary
  - Best hyperparameters (hidden_size, latent_dim, activation, optimizer, epochs, k, lr, batch_size, etc.)
  - Random seed
  - Input dimension

**Usage:**
```python
checkpoint = torch.load('D1_Swiss_processed_model.pth')
model = Autoencoder(INPUT_DIM, checkpoint['best_hidden_size'], 
                    checkpoint['best_latent_dim'], activation_fn)
model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. H3 Validation Data (Pickle)
- **File:** `h3_data.pkl`
- **Contents:**
  - `cluster_labels_all`: Train+val cluster assignments
  - `test_cluster_assignments`: Test cluster assignments
  - `cluster_centroids`: Cluster centroids
  - `best_k`: Optimal number of clusters
  - `best_latent_dim`: Optimal latent dimension
  - `RANDOM_SEED`: Random seed for reproducibility
  - `dataset_name`: Dataset name
  - `profile_summary`: Profile characteristics

**Usage:**
```python
import pickle
with open('h3_data.pkl', 'rb') as f:
    h3_data = pickle.load(f)
```

### 3. H3 Validation Data (NumPy)
- **File:** `h3_data.npz`
- **Contents:** Same as pickle, but in NumPy format (more space-efficient)
- **Usage:**
```python
import numpy as np
h3_data = np.load('h3_data.npz', allow_pickle=True)
```

### 4. Full Pipeline Results
- **File:** `pipeline_results.pkl`
- **Contents:**
  - All data from h3_data.pkl
  - `train_val_data`: Original 4D symptom data
  - `latent_vectors_all`: Latent representations
  - `final_sil_score`, `final_ch_score`, `final_db_score`: Clustering metrics
  - `reconstruction_loss`: Test reconstruction loss

---

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Early H3 Check                                           │
│    - Verify dataset has PSYT_Therapy_Use column            │
│    - Verify it's a Swiss dataset                           │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Data Preparation                                         │
│    - Load D1_Swiss_processed.csv                           │
│    - Extract 4 features: [Depression, Anxiety, Stress,      │
│      Burnout]                                               │
│    - Stratified 80/20 split (train+val / test)            │
│    - Create 10-fold CV splits                              │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Stage 1: Architecture Tuning (2,700 experiments)        │
│    - Grid search: hidden_size, latent_dim, activation,     │
│      optimizer, epochs                                      │
│    - 10-fold CV for each config                             │
│    - Consensus voting for K selection                      │
│    - Select best architecture                               │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Stage 2: Learning Parameters (~270 experiments)          │
│    - LR Range Test                                          │
│    - Grid search: batch_size, weight_decay, momentum       │
│    - 10-fold CV                                             │
│    - Select best learning parameters                        │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 5. Final Model Training                                      │
│    - Train on all train+val data                           │
│    - Extract latent vectors                                 │
│    - K-means clustering (optimal K)                        │
│    - Extract profile characteristics                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 6. Test Set Evaluation                                       │
│    - Encode test data                                       │
│    - Assign to clusters (using trained centroids)          │
│    - Evaluate metrics                                       │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 7. H3 Validation                                            │
│    - Combine all cluster assignments                       │
│    - Align with therapy labels                             │
│    - Chi-square test                                        │
│    - Cramér's V + residuals                                │
│    - Conclusion                                             │
└─────────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────────┐
│ 8. Save Results                                             │
│    - Model (.pth)                                           │
│    - H3 data (.pkl, .npz)                                  │
│    - Full results (.pkl)                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### Consensus Voting for K Selection
- Tests K=[2,3,4,5,6] for each configuration
- 4 methods vote: Silhouette, Calinski-Harabasz, Davies-Bouldin, Elbow
- If 2+ methods agree → use that K
- If no consensus → use Silhouette (most interpretable)

### Profile Characteristics
- Mean symptom levels (Depression, Anxiety, Stress, Burnout) per cluster
- Computed from **original 4D symptom space**, not latent space
- Used for interpretation and replication testing

### H3 Validation Logic
- **Null Hypothesis (H0):** Profile membership is independent of therapy use
- **Alternative (H1):** Profile membership is associated with therapy use
- **Test:** Chi-square test of independence
- **Effect Size:** Cramér's V (strength of association)
- **Post-hoc:** Standardized residuals (which profiles deviate)

---

## Troubleshooting

### Issue: H3 validation skipped
**Cause:** Dataset name mismatch or missing column
**Fix:** Ensure dataset name is `D1_Swiss_processed` and CSV has `PSYT_Therapy_Use` column

### Issue: Slow execution
**Cause:** Running on CPU or too many parameters
**Fix:** 
- Use GPU (A100 recommended)
- Reduce `n_folds` to 5
- Reduce parameter ranges

### Issue: Out of memory
**Cause:** Batch size too large or too many experiments
**Fix:** Reduce batch sizes in Stage 2 grid search

### Issue: Model collapse (all latent vectors identical)
**Cause:** Learning rate too high or architecture issue
**Fix:** Check LR Range Test results, adjust learning rate

---

## Expected Outputs

### Console Output
- Progress bars for each stage
- Best configuration summaries
- Profile characteristics table
- Test set metrics
- H3 validation results (contingency table, chi-square, p-value, Cramér's V, residuals)
- H3 conclusion (VALIDATED or NOT VALIDATED)

### Files Created
```
D1_Swiss_processed_results/
├── D1_Swiss_processed_model.pth
├── h3_data.pkl
├── h3_data.npz
└── pipeline_results.pkl
```

---

## Next Steps After Pipeline

1. **Load saved model** for inference on new data
2. **Use H3 results** for clinical interpretation
3. **Compare profiles** across datasets (replication testing)
4. **Visualize** latent space and profile characteristics

---

## Notes

- **Reproducibility:** Random seed = 42 (set at top of script)
- **Stratification:** Uses optimal bin numbers for balanced train/test splits
- **Early stopping:** H3 check prevents wasted training time
- **Model saving:** All hyperparameters saved for exact reproduction

---

**Last Updated:** Based on `d3_plswork.py` configuration
**Dataset:** D1_Swiss_processed (~800 samples)
**CV Folds:** 10-fold
**Expected Runtime:** ~1.5-2 hours on A100 GPU

