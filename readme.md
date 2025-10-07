# Constraint-Aware Counterfactual Explanations

This repository contains the implementation for generating constraint-aware counterfactual explanations using denial constraints and SMT solvers, as described in our paper.

## Overview

This codebase provides methods for:

- **Projection**: Projecting infeasible instances onto the constraint-satisfying manifold
- **Counterfactual Generation**: Creating feasible, diverse counterfactual explanations for machine learning models
- **Constraint Handling**: Supporting both unary and binary denial constraints

## Key Features

- **Solver-Based Projection**: Two variants (Single Solver with preprocessing, Suspect Set without preprocessing)
- **Linear Model Integration**: Direct encoding of linear decision boundaries into the solver
- **Neural Network Support**: Perturb-and-project approach for neural networks
- **Multiple Baselines**: Exhaustive search, best-in-dataset selection
- **Comprehensive Metrics**: Proximity (MAD, L0, L1), diversity (DPP, pairwise, minimum), constraint violations

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=1.9.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
dice-ml>=0.9
z3-solver>=4.8.12
matplotlib>=3.4.0
```

## Repository Structure

```
.
├── perturb_test.py          # Main script for CF generation with projection
├── projection_test.py       # Projection-only experiments
├── evaluate.py              # Evaluation metrics computation
├── perturb.py              # Core projection algorithms
├── eval.py                 # Evaluation utility functions
├── class_models.py         # Neural network models
├── scripts/                # Example scripts per dataset
│   ├── adult_*.sh
│   ├── ny_*.sh
│   ├── census_*.sh
│   └── tax_*.sh
├── data/
│   ├── datasets/           # CSV datasets
│   └── constraints/        # Denial constraint files
└── README.md
```

## Quick Start

### 1. Projection Only

Test projection algorithms without CF generation:

```bash
python projection_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 --k_upper 6 \
    --num_samples 10 \
    --exp_name adult_projection_test \
    --projection_mode solver
```

### 2. Counterfactual Generation

Generate constraint-aware counterfactuals:

```bash
python perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 --k_upper 6 \
    --num_samples 10 \
    --exp_name adult_neural_cfs \
    --delta 50 \
    --solver_timeout 10000
```

### 3. Linear Model Integrated Approach

For linear models with integrated constraint handling:

```bash
python perturb_test.py \
    --cont_feat age education_num hours_per_week \
    --fixed_feat age race sex \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --k_lower 5 --k_upper 6 \
    --linear_model \
    --exp_name adult_linear_integrated \
    --delta 50 \
    --solver_timeout 20000
```

## Parameter Reference

### Core Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `--cont_feat` | list | Continuous feature names | `age education_num hours_per_week` |
| `--fixed_feat` | list | Immutable features (e.g., age, race, sex) | `age race sex` |
| `--dataset_path` | str | Path to CSV dataset | `data/datasets/adult.csv` |
| `--constraints_path` | str | Path to denial constraints file | `data/constraints/adult_dcs.txt` |
| `--k_lower` | int | Minimum number of CFs to generate | `5` |
| `--k_upper` | int | Maximum number of CFs to generate | `6` |
| `--num_samples` | int | Number of test instances | `10` |
| `--exp_name` | str | Experiment name for output directory | `adult_experiment` |

### Model Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--epochs` | int | Training epochs for neural network | `10` |
| `--load_model` | flag | Load pre-trained model | `False` |
| `--linear_model` | flag | Use linear SVM classifier | `False` |
| `--linear_pandp` | flag | Use perturb-and-project with linear model | `False` |

### Solver Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `--solver_timeout` | int | Solver timeout in milliseconds | `10000` |
| `--timeout` | int | Projection timeout in seconds | `1000` |
| `--delta` | float | Diversity weight parameter | `0.5` |
| `--fixed_flag` | flag | Reset solver cache between projections | `False` |

### Projection Mode

| Parameter | Type | Description | Options |
|-----------|------|-------------|---------|
| `--projection_mode` | str | Projection algorithm | `solver`, `exhaustive`, `best_in_dataset` |

**Options:**
- `solver`: SMT solver-based projection (recommended)
- `exhaustive`: Exhaustive search (only for small datasets)
- `best_in_dataset`: Select closest tuple from dataset

## Dataset Format

### CSV Dataset Structure

```csv
age,education_num,hours_per_week,workclass,education,...,label
39,13,40,State-gov,Bachelors,...,0
50,13,13,Self-emp-not-inc,Bachelors,...,0
```

- Last column must be named `label` (0 or 1 for binary classification)
- Categorical features will be auto-detected
- Continuous features specified via `--cont_feat`

### Denial Constraints Format

Constraints use the format: `¬{condition1 ∧ condition2 ∧ ...}`

Example (`adult_dcs.txt`):

```
t0.relationship == "Wife" && t0.sex != "Female"
t0.age < 18 && t0.hours_per_week > 40
t0.education_num >= t1.education_num && t0.income < t1.income
```

- `t0.` refers to the counterfactual instance
- `t1.` refers to database tuples (for binary constraints)
- Operators: `==`, `!=`, `<`, `<=`, `>`, `>=`

## Example Scripts

See the `scripts/` directory for complete example scripts for each dataset:

### Adult-Income Dataset

```bash
# Neural network with solver projection
bash scripts/adult_neural_solver.sh

# Linear model integrated approach  
bash scripts/adult_linear_integrated.sh

# Projection-only experiments
bash scripts/adult_projection_only.sh
```

### NY-Housing Dataset

```bash
# Neural network experiments
bash scripts/ny_neural_solver.sh

# Linear model experiments
bash scripts/ny_linear_integrated.sh
```

### Census-Income Dataset

```bash
# Large-scale experiments (7M+ assertions)
bash scripts/census_neural_solver.sh
bash scripts/census_linear_integrated.sh
```

### Tax Dataset

```bash
# Synthetic dataset experiments
bash scripts/tax_neural_solver.sh
bash scripts/tax_linear_integrated.sh
```

## Output

Results are saved in `data/{exp_name}/`:

- `dice_cfs_sample{i}_k{k}.csv` - Original DiCE counterfactuals
- `solver_pandp_cfs_sample{i}_k{k}.csv` - Projected counterfactuals
- `solver_linear_cfs_sample{i}_k{k}.csv` - Linear integrated CFs
- `commandline_args.txt` - Command-line arguments used
- `ml_model_state_dict.pth` - Trained model weights

## Evaluation

Run evaluation to compute metrics:

```bash
python evaluate.py \
    --dataset_path data/datasets/adult.csv \
    --constraints_path data/constraints/adult_dcs.txt \
    --dataset adult_experiment \
    --k_lower 5 --k_upper 6 \
    --mode solver
```

### Metrics Computed

**Proximity Metrics:**
- MAD-normalized distance
- L0 distance (sparsity)
- L1 distance

**Diversity Metrics:**
- DPP (Determinantal Point Process)
- Pairwise diversity
- Minimum distance diversity

**Constraint Violations:**
- Total violations per CF
- Unary constraint violations
- Tuple conflicts (binary constraints)

## Algorithm Variants

### 1. Single Solver (Preprocessing)

- **Use when:** Multiple projections needed, can amortize preprocessing cost
- **Advantages:** Fastest per-instance runtime after preprocessing
- **Set:** Default mode, preprocessing happens automatically

### 2. Suspect Set (No Preprocessing)

- **Use when:** Few projections needed, want zero upfront cost
- **Advantages:** No preprocessing required, filters constraint space
- **Set:** Use same code, filtering happens automatically based on fixed features

### 3. Linear Integrated

- **Use when:** Using linear classifiers (SVM, LogisticRegression)
- **Advantages:** Superior quality, joint optimization
- **Set:** Add `--linear_model` flag

### 4. Perturb-and-Project (P&P)

- **Use when:** Using neural networks
- **Advantages:** Works with any black-box model
- **Set:** Default for neural networks (omit `--linear_model`)

## Reproducing Paper Results

To reproduce the experimental results from the paper:

### Projection Experiments (Figure 4, Figure 5)

```bash
# Run all projection experiments
bash scripts/run_all_projection.sh
```

### Neural Network CFs (Figure 6, Table 5, Table 6)

```bash
# Run all neural network CF experiments  
bash scripts/run_all_neural_cfs.sh
```

### Linear Model CFs (Figure 7, Table 7, Table 8)

```bash
# Run all linear model CF experiments
bash scripts/run_all_linear_cfs.sh
```

## Troubleshooting

### Out of Memory Errors

For large datasets (Census), reduce batch size or use Suspect Set variant:

```bash
# Already uses Suspect Set automatically
python perturb_test.py --dataset_path data/datasets/census.csv ...
```

### Solver Timeouts

Increase solver timeout for complex constraint spaces:

```bash
python perturb_test.py --solver_timeout 50000 ...  # 50 seconds
```

### Slow Projection

Enable solver cache reset to free memory:

```bash
python perturb_test.py --fixed_flag ...
```

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{yourpaper2024,
  title={Constraint-Aware Counterfactual Explanations with Denial Constraints},
  author={Your Name and Coauthors},
  booktitle={Conference Name},
  year={2024}
}
```

## License

[Your License Here]

## Contact

For questions or issues, please open a GitHub issue or contact [your-email@domain.com].