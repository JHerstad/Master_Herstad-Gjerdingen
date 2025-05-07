# Master_Herstad-Gjerdingen

MASTER_HERSTAD-GJERDINGEN/
├── config/                         # Configuration files
│   ├── config.py                  # Centralized configuration class for the pipeline
│   └── defaults.py                # Default settings for the project
├── data/                          # Datasets and experiment outputs
│   ├── Aachen/                    # Aachen battery degradation dataset
│   ├── MIT_Stanford/              # MIT Stanford dataset (optional)
│   ├── hyperparameters_tuning/    # Hyperparameter tuning results
│   ├── models/                    # Trained models
│   └── results_XAI/               # Explainability results
├── notebooks/                     # Jupyter notebooks for experimentation
│   ├── 01_preprocessing.ipynb    # Preprocess datasets
│   ├── 02_grid_search.ipynb      # Perform hyperparameter tuning
│   ├── 03_model_training.ipynb   # Train LSTM/CNN models
│   ├── 04_model_evaluation.ipynb # Evaluate model performance
│   └── Explainable_AI_copy.ipynb # Explore explainability (work in progress)
├── requirements/                  # Dependency requirements
│   └── timeshap_env_requirements.txt  # Environment requirements for TimeSHAP explainability
├── src/                           # Python scripts for reusable code
│   ├── evaluation.py             # Model evaluation functions
│   ├── grid_search.py            # Hyperparameter tuning logic
│   ├── models.py                # Model definitions (LSTM, CNN, etc.)
│   ├── preprocessing.py         # Data preprocessing functions
│   └── utils.py                # Utility functions
├── Testing_New_Dataset/          # Experiments with new datasets (optional)
├── .gitignore                    # Git ignore file
└── README.md                     # Project documentation
