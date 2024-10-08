## Dataset Directory

The datasets used for experiments are located in the `data/` directory. You can create subdirectories for different datasets, e.g., `data/dataset_name/`, and place your dataset files there.

## Model Code Directory

The model implementation can be found in the `models/` directory. The main file, `model.py`, contains the architecture and training routines for our dual-channel graph-level anomaly detection method.

## Main Content and Modules

- **Data Preprocessing**: The `utils/preprocess.py` file includes functions to preprocess the input data into the required format for the model.
- **Model Training**: The `models/model.py` contains the implementation of the dual-channel graph-level model, including the training and evaluation procedures.
- **Experiment Configuration**: The `experiments/config.py` file allows users to set various parameters for running experiments.
- **Running Experiments**: The `experiments/run_experiment.py` script is provided to facilitate the execution of experiments with the specified configuration.

## Illustrations

![Anomaly Detection Framework](dacd.png)  
*Figure 1: Overview of the dual-channel anomaly detection framework.*

## Citation


