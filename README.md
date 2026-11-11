# Power-Domination ML

This repository contains the datasets and source code supporting the study:

**Structural Predictors of Power-Domination Propagation Dynamics in Complex Networks: An Explainable Machine-Learning Approach**

## Repository structure

```text
power-domination-ml/
├── code/
├── data/
├── LICENSE
└── README.md
```

## Data

The `data/` directory contains the datasets used in the analyses reported in the study.

The primary dataset is:

- `pds_propagation_final_results.csv`

This dataset contains the graph- and candidate-set-level features and outcomes used for the main machine-learning analyses.

Additional datasets include:

- `extended_network_dataset.csv`
- `parameter_sensitivity_dataset.csv`

These datasets support the additional generalizability and parameter-sensitivity analyses.

## Code

The `code/` directory contains the MATLAB and Python source code used for:

- synthetic network generation
- candidate-set generation
- power-domination simulation
- structural feature extraction
- Model A classification
- Model B regression
- repeated graph-level validation
- strategy-balanced sensitivity analysis
- cross-topology transfer analysis
- distribution-shift analysis
- parameter-sensitivity analysis
- structural-feature robustness analysis
- reproducibility and uncertainty analyses
- runtime benchmarking
- generation of reported tables and figures

The MATLAB helper functions used by the simulation workflow are located in:

```text
code/functions/
```

## Machine-learning framework

### Model A

Model A predicts whether a candidate set successfully forms a power-dominating set.

### Model B

Model B predicts power propagation time (PPT) conditional on successful power domination.

Accordingly, Model B is fitted only to successful power-domination configurations.

## Reproducibility

The analyses use graph-level data splitting to avoid information leakage between training and test sets.

The primary random seed used in the machine-learning analyses is:

```text
2026
```

Repeated holdout validation was performed using the predefined random seeds reported in the manuscript.

## Software

The analyses were conducted using MATLAB and Python.

The main Python dependencies include:

- NumPy
- pandas
- NetworkX
- scikit-learn
- SciPy
- Matplotlib
- SHAP

Exact software versions can be added in a `requirements.txt` file if required.

## Data and code availability

The datasets and source code supporting the study are publicly available in this repository.

A permanent archived version of this repository will also be deposited in Zenodo and assigned a DOI.

## Citation

If you use the data or code from this repository, please cite the associated article and the Zenodo archive.

The full article citation and Zenodo DOI will be added following publication and repository archiving.

## License

The source code is distributed under the MIT License.
