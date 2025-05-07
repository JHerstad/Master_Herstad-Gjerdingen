# Master_Herstad-Gjerdingen


# Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction

**Spring 2025 Master's Thesis**  
*Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction*

This repository was developed as part of the Master’s thesis “Explainable Time-Series Modeling for Battery Remaining Useful Life Prediction.” It contains code to reconstruct all experiments comparing post-hoc explanations of a baseline black-box time-series battery model against intrinsically interpretable models, in terms of both explainability and predictive performance.

## Table of Contents

- [Project Structure](#project-structure)  
- [Datasets](#datasets)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Modules](#modules)  
- [Experiment Workflow](#experiment-workflow)  
- [Contributing](#contributing)  
- [License](#license)  
- [Authors](#authors)  

## Project Structure

```text
MASTER_HERSTAD-GJERDINGEN/
├── config/                         # Configuration files
│   ├── config.py                  # Centralized configuration class for the pipeline
│   └── defaults.py                # Default settings for the project
├── data/                           # Datasets and experiment outputs
│   ├── Aachen/                     # Aachen battery degradation dataset
│   └── MIT_Stanford/               # MIT Stanford dataset (optional)
├── experiments/                    # Experiment outputs
│   ├── hyperparameters_tuning/     # Hyperparameter tuning results
│   ├── models/                     # Trained models
│   ├── results/                    # Raw and processed results
│   └── results_XAI/                # Explainability results
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── 01_preprocessing.ipynb      # Preprocess datasets
│   ├── 02_grid_search.ipynb        # Perform hyperparameter tuning
│   ├── 03_model_training.ipynb     # Train LSTM/CNN models
│   ├── 04_model_evaluation.ipynb   # Evaluate model performance
│   └── Explainable_AI_copy.ipynb   # Explore explainability (work in progress)
├── requirements/                   # Dependency requirements
│   └── timeshap_env_requirements.txt  # Environment requirements for TimeSHAP explainability
├── src/                            # Python scripts for reusable code
│   ├── evaluation.py               # Model evaluation functions
│   ├── grid_search.py              # Hyperparameter tuning logic
│   ├── models.py                   # Model definitions (LSTM, CNN, etc.)
│   ├── preprocessing.py            # Data preprocessing functions
│   └── utils.py                    # Utility functions
├── Testing_New_Dataset/            # Experiments with new datasets (optional)
├── .gitignore                      # Git ignore file
└── README.md                       # Project documentation (this file)
```

## Datasets

Place the raw datasets under the following directories:

- **`data/Aachen/raw`**: Contains the raw Aachen battery degradation dataset.

The preprocessed datasets used in the experiments are stored in:

- **`data/Aachen/processed`**: Preprocessed Aachen dataset.  
- **`data/MIT_Stanford/processed`**: Preprocessed MIT Stanford dataset.

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

## Usage

### Preprocessing

```bash
python src/preprocessing.py --dataset Aachen --config config/defaults.py
```

### Hyperparameter Tuning

```bash
python src/grid_search.py --dataset Aachen --config config/config.py
```

### Model Training

```bash
python src/models.py --train --dataset Aachen --config config/config.py
```

### Evaluation

```bash
python src/evaluation.py --model-path data/models/<model_name>.pt --dataset Aachen
```

### Explainable AI Generation

Run the notebook `notebooks/Explainable_AI_copy.ipynb` or use the command-line interface:

```bash
python src/utils.py --generate-xai --model-path data/models/<model_name>.pt --dataset Aachen
```

## Modules

- **config/**: Centralized configuration management.  
- **src/preprocessing.py**: Data cleaning and feature extraction.  
- **src/grid_search.py**: Automated hyperparameter search.  
- **src/models.py**: Definition and training routines for LSTM and CNN models.  
- **src/evaluation.py**: Performance metrics and result visualization.  
- **src/utils.py**: Utility functions including XAI generation.

## Experiment Workflow

1. Preprocess the raw data.  
2. Perform hyperparameter tuning.  
3. Train final models.  
4. Evaluate performance on held-out test sets.  
5. Generate post-hoc explanations and compare against interpretable baselines.

## Contributing

Contributions are welcome! Please open issues or submit pull requests following the project’s [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Authors

- **Sigurd Herstad-Gjerdingen** (<your_email@domain.com>)  
- **Johannes Øen Herstad** 