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
│   ├── master_env_requirements.txt
│   └── master_env.yml             
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

The MIT/Stanford dataset requires downloading raw data and specific preprocessing steps, while the Aachen dataset can be processed directly. Both "raw" datasets are preprocessed to be univariate. To configure with other preprocessing parameters, change them (e.g., EOL80 or EOL65) in `defaults.py` 

### MIT/Stanford Dataset
- **Data Source**: Download the raw batch data from [https://data.matr.io/1/](https://data.matr.io/1/), where additional details about the dataset are available.
- **Preprocessing Instructions**:
  1. Run the `BuildPkl_Batch1.ipynb` notebook, specifying the file path for each batch (e.g., `2017-05-12_batchdata_updated_struct_errorcorrect.mat` for Batch 1, and similar paths for Batches 2 and 3).
  2. This generates `batch.pkl` files, saved in the `MIT_Stanford/raw` directory.
  3. Execute the `01_preprocessing.ipynb` notebook to complete the preprocessing.

### Aachen Dataset
- **Preprocessing Instructions**: Run `01_preprocessing.ipynb`. No raw dataset files need to be downloaded or placed.

## Experiment Workflow

### Preprocess the Raw Data
This step is performed by running the `01_preprocessing.ipynb` notebook.

The notebook preprocesses both the Aachen and MIT/Stanford datasets for regression and classification tasks, setting End-of-Life (EOL) capacity thresholds

### Perform Hyperparameter Tuning
This step is done by running the `02_grid_search.ipynb` notebook.

The notebook conducts hyperparameter tuning for multiple models (LSTM, CNN, Decision Tree, Linear Regression, Lasso Regression, Logistic Regression) on both Aachen and MIT/Stanford datasets, using grid search or hyperparameter search. It logs the best hyperparameters and performance metrics (e.g., validation loss, cross-validation accuracy) for each model.

### Train Final Models
This step is completed by running the `03_model_training.ipynb` notebook.

The notebook trains final models (LSTM, CNN, Decision Tree, Linear Regression, Lasso Regression, Logistic Regression) on the preprocessed Aachen and MIT/Stanford datasets using the best hyperparameters from the tuning step. It logs training progress and visualizes training history (e.g., loss and MAE over epochs) for deep learning models like LSTM and CNN.

### Evaluate Performance on Held-Out Test Sets
This step is executed by running the `04_model_evaluation.ipynb` notebook.

The notebook logs evaluation metrics (RMSE, MAE, R² for regression; accuracy, F1-score, precision, recall for classification) and generates visualizations (prediction plots for regression, confusion matrix for classification) to assess model performance comprehensively.

### Explainable AI 
This step is executed by running the `Explainable_AI.ipynb` notebook.

This notebook serves as the primary workflow for the project. It begins by loading the trained models, followed by generating ante-hoc explanations for intrinsically interpretable models. Next, post-hoc explanations (LIME and SHAP) are produced and compared for the LSTM model, while explanations (LIME, SHAP, and Grad-CAM) are generated and compared for the CNN model. Finally, the explanations are quantitatively evaluated using faithfulness metrics (PGI and PGU) and stability metrics (RIS and ROS).


## Installation

1. Clone this repository:  
   ```bash
   git clone https://github.com/<your_org>/MASTER_HERSTAD-GJERDINGEN.git
   cd MASTER_HERSTAD-GJERDINGEN
   ```
2. Create and activate a virtual environment. Choose one of the following options:  

   **Option A: Using `venv` (pip-based)**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   pip install -r requirements/master_env_requirements.txt
   ```

   **Option B: Using Conda**  
   ```bash
   conda env create -f requirements/master_env.yml
   conda activate master_env
   ```

## Authors

- **Sigurd Gjerdingen**
- **Johannes Herstad**
