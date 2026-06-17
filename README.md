# Uplift-Modeling-RecSys

The repository contains code and materials for the coursework *"Uplift Modeling in Recommender Systems"*. This project combines propensity score estimation, relevance modeling, and causal effect estimation to improve recommendation quality while addressing selection bias. The basis for the project realisation is taken from [PropCare](https://github.com/mediumboat/PropCare) by [mediumboat](https://github.com/mediumboat).

## Overview

This system implements multiple causal recommendation approaches including:
- **DLCE** with various causal metrics
- **PropCare** for propensity score estimation
- **Causal effect estimation** through IPS (Inverse Propensity Scoring) and DR (Doubly Robust) methods
- Multiple causal evaluation metrics (DIR@K, RCAU@K, CPrecS, CDCGS)

## Features

- **Multiple Recommender Variants**:
  - `DLMF`: Basic DLCE with various final predictions (standard, frequency or relevance estimates influenced)
  - `DLMF_Mod`: Extended version with separate treatment and relevance effects
  - Standard baselines (Popularity, Random, MF)
- **Evaluation Metrics**:
  - Causal metrics: CP@K, CDCG, DIR@K, RCAU@K
  - Traditional metrics: NDCG@K, Recall@K, Precision@K
- **Flexible Dataset Support**: Dunn Cate, ML-100k, Finn-No, T-ECD

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

### Key Dependencies

- TensorFlow 2.19.0+
- TensorFlow Probability
- Scikit-learn
- Pandas, NumPy
- Hugging Face Hub (for T-ECD dataset)
- tqdm, matplotlib

## Project Structure

```
.
├── main.py              # Main execution script
├── train.py             # Training pipeline and data preparation
├── models.py            # PropCare code
├── baselines.py         # Recommender implementations
├── evaluator.py         # Evaluation metrics
├── tecd_downloader.py   # T-ECD dataset downloader
├── requirements.txt     # Dependencies
└── results/             # Output directory (created at runtime)
```

## Usage

### Basic Usage

```bash
python main.py --dataset t-ecd-small-short-m
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset to use (1d, 10d, 1p, 10p, ml, f, t-ecd-small-*) | 'd' |
| `--prop_type` | Propensity model type (orig, mod) | 'mod' |
| `--rec_type` | Recommender type (orig, rel, mod, dr, gbc) | 'orig' |
| `--batch_size` | Batch size | 5096 |
| `--repeat` | Number of runs | 1 |
| `--prop_train` | Train propensity model | False |
| `--rec_train` | Train recommender | False |
| `--continue_rec_train` | Continue training from checkpoint | False |

### Supported Datasets

**T-ECD** (t-ecd-small-short-{m,r}, t-ecd-small-long-{m,r})
   - E-commerce transaction data
   - Marketplace (m) and Retail (r) domains
   - Short and long timeframes

### T-ECD Dataset Download

```python
from tecd_downloader import download_dataset

download_dataset(
    token="your_hf_token",
    local_dir="t_ecd_small",
    domains=["retail", "marketplace"],
    day_begin=1223,
    day_end=1308
)
```

## Output

Results are saved in the `results/` directory with the following structure:
```
results/
├── result_{dataset}.txt          # Final aggregated results
└── {add}/                        # Default: 'default'
    └── {prop_type}/              # 'orig' or 'mod'
        └── {dataset}/            # e.g., 'm', 'r', etc.
            ├── {dataset_version}/ # Optional: 'short/' or 'long/' for T-ECD
            │   └── {num_run}_prop.weights.h5      # Actually saved as {prop_add}
            └── {rec_type}/        # 'orig', 'rel', 'mod'
                └── {dataset_version}/ # Optional
                    ├── {num_run}_dlmf_weights.pkl      # For rec_type='orig' or 'rel'
                    └── {num_run}_dlmf_mod_weights.pkl  # For rec_type='mod'
```

The output file contains all evaluation metrics averaged across runs.

## Acknowledgments

- The T-ECD dataset is from the T-ECH team on Hugging Face
- The basis for the project realisation is taken from [PropCare](https://github.com/mediumboat/PropCare) by [mediumboat](https://github.com/mediumboat)