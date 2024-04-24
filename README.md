# DLNLP_assignment_24_SN23047341
## PII Data Detection
This project aims to address the critical task of Personally Identifiable Information (PII) detection within text data. The core of the system is built on the robust DeBERTa model, renowned for its powerful NLP capabilities. To tailor DeBERTa for the specific nuances of PII detection, we employ Low-Rank Adaptation (LoRA), an efficient and effective method for fine-tuning large-scale transformer models.

LoRA introduces trainable low-rank matrices that adapt the self-attention and feed-forward networks of pre-trained transformers, such as DeBERTa, allowing us to modify the model behavior with minimal updates and increased training efficiency.

The project is structured to be modular and extensible, with a collection of Python scripts handling different stages of the machine learning pipeline, from data preprocessing and exploratory analysis to model training, evaluation, and inference.

Our approach focuses on achieving high precision and recall in identifying sensitive information, ensuring that the resulting model can be reliably used to protect privacy in a variety of applications.


## Project Structure

Below is the directory layout for this project:

- `Dataset/`: Contains datasets and related files for training and evaluation.

- `func/`: Helper functions for various tasks including data preparation, evaluation, and training procedures.
  - `__init__.py`: Indicates that this directory is a Python package.
  - `EDA.py`: Script for Exploratory Data Analysis.
  - `find_spans.py`: Utility to find and process spans in text data.
  - `func_for_plot.py`: Functions to generate plots.
  - `get_data.py`: Functions to load and preprocess data.
  - `global_seed.py`: Utility to set the seed for reproducibility.
  - `metrics.py`: Metrics for evaluating the model's performance.
  - `tokenize.py`: Custom tokenization functions.

- `image_output/`: Directory for output images such as plots and figures.
  - `class.pdf`: Visual representation of class distributions with names.
  - `class_without_name.pdf`: Visual representation of class distributions without names.
  - `distribution.pdf`: Distribution plot for data analysis.

- `logs/`: Contains logs generated during training and evaluation.

- `model/`: Code related to the model architecture.
  - `__init__.py`: Indicates that this directory is a Python package.
  - `model.py`: Defines the model architecture and functions.

- `run/`: Scripts to run the model training and testing.
  - `checkpoint/`: Stores checkpoints during model training.
  - `test.py`: Script for model testing.
  - `train.py`: Script for model training.

- `main.py`: Main script to run the entire pipeline, orchestrating training and testing.

## Main Script Arguments

In `main.py`, the following command-line arguments can be used to configure the training and evaluation processes. Each option comes with a default value, but you can specify your own values to customize the behavior as needed.

- `--training_max_length` (default: `2048`): The maximum sequence length for training input data.
- `--eval_max_length` (default: `3072`): The maximum sequence length for evaluation input data.
- `--lr` (default: `5e-4`): The learning rate for the optimizer.
- `--lr_scheduler_type` (default: `"linear"`): Type of learning rate scheduler used during training.
- `--num_epochs` (default: `3`): The number of epochs to train the model.
- `--batch_size` (default: `1`): The batch size used for training.
- `--eval_batch_size` (default: `1`): The batch size used during evaluation.
- `--warmup_ratio` (default: `0.1`): The proportion of training to warm up the learning rate.
- `--weight_decay` (default: `0.01`): The weight decay to apply to the weights of the model.
- `--freeze_layers` (default: `6`): The number of initial layers of the transformer to freeze during training.
- `--lora_r` (default: `16`): The rank of the LoRA (Low-Rank Adaptation) layers.
- `--amp` (default: `True`): Whether to use automatic mixed precision for training.
- `--n_splits` (default: `4`): The number of data splits used for cross-validation.
- `--negative_ratio` (default: `0.3`): The ratio of negative samples to positive samples in the dataset.
- `--output_dir` (default: `"output"`): The directory where outputs such as the trained model and logs will be saved.
- `--data_folder` (default: `"Dataset"`): The directory containing the training and evaluation data files.



## Packages and Requirements

This program runs under Python version 3.10.
The project depends on several external libraries, which are listed in `environment.yml`. To install these dependencies, run the command below:
```sh
conda env create -f environment.yml
```



## Example Run Command

To start training with the default parameters, run the following command in the terminal:
```bash
python main.py
```
or you can choose your own parameters like this:
```bash
python main.py --training_max_length 1024 --num_epochs 5
```

