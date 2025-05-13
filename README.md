# Master_Herstad-Gjerdingen


# Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction

**Spring 2025 Master's Thesis**  
*Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction*

This repository was developed as part of the Master’s thesis “Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction.” It contains code to reconstruct all experiments comparing post-hoc explanations of a baseline black-box time-series battery model against intrinsically interpretable models, in terms of both explainability and predictive performance.

## Table of Contents

- [Project Structure](#project-structure)  
- [Datasets](#datasets)  
- [Installation](#installation)  
- [Experiment Workflow](#experiment-workflow)  
- [Authors](#authors)  

## Project Structure

```text
MASTER_HERSTAD-GJERDINGEN/
├── config/                         
│   └── defaults.py                 # Default settings for the project
├── data/                           # Datasets and experiment outputs
│   ├── Aachen/                     
│   └── MIT_Stanford/               
├── experiments/                    # Experiment outputs
│   ├── hyperparameters_tuning/     # Hyperparameter tuning results
│   ├── models/                     # Pre-Trained models
│   ├── results/                    # Evaluation Metric Results, Intrinsic Model Parameters
│   └── results_XAI/                # Post-Hoc Explainer results
├── notebooks/                      
│   ├── 01_preprocessing.ipynb      
│   ├── 02_grid_search.ipynb        
│   ├── 03_model_training.ipynb     
│   ├── 04_model_evaluation.ipynb   
│   └── Explainable_AI.ipynb        # Main Notebook
├── requirements/                   # Dependency requirements
│   ├── master_env_requirements.txt # Pip-compatible requirements
│   └── master_env.yml             # Conda environment YAML
├── src/                           # Python functions used in notebooks
│   ├── evaluation.py               
│   ├── grid_search.py              
│   ├── models.py                   
│   ├── preprocessing.py            
│   └── utils.py                    
├── .gitignore                      
└── README.md                       
```

## Datasets

Only preprocessed versions of datasets are available in this repository due to size constraints.

To redo the preprosessing, place the raw datasets under the following directories, set parameters (e.g. EOL80/EOL65) in defaults.py and run the 01_preprocessing.ipynb notebook.


## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/<your_org>/MASTER_HERSTAD-GJERDINGEN.git
   cd MASTER_HERSTAD-GJERDINGEN
   ```
2. Create and activate a virtual environment (recommended):  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:  
   ```bash
   pip install -r requirements/timeshap_env_requirements.txt
   ```

## Experiment Workflow

1. Preprocess the raw data.  
2. Perform hyperparameter tuning.  
3. Train final models.  
4. Evaluate performance on held-out test sets.  
5. Generate post-hoc explanations and compare against interpretable baselines.

## Authors

- **Sigurd Gjerdingen**
- **Johannes Herstad**
