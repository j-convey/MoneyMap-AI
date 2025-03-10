# MoneyMap-AI

**MoneyMap-AI** is an AI-powered tool that categorizes bank transactions into categories and subcategories using a BERT-based model. It helps users organize financial data efficiently through training and prediction workflows, accessible via a command-line interface (CLI).

## Prerequisites
-  Python 3.9 or later
- Required dependencies (see Installation)
- NVIDIA GPU with CUDA support (optional, for faster training; CPU is supported)

## CLI Usage
## General Syntax
Run the main.py script with one of two commands: train or predict. Each command requires a path argument specifying the data file (for training) or model file (for prediction).


```python main.py <command> <path>```
- <command>: Either train or predict.
- <path>: The file path to the training data (for train) or the pre-trained model (for predict).

1. Training a Model
Use the train command to train the model on your bank transaction data.
Example:
```python main.py train data/main.csv```


2. Making Predictions
Use the predict command to classify transactions using a pre-trained model.

Example:
```python main.py predict /path/to/models/pt_cat_modelV1```




## Dependencies
The project relies on the following Python libraries:
- torch: PyTorch framework for deep learning.
- torch.nn and torch.optim: Neural network and optimization modules from PyTorch.
- matplotlib: For plotting training history.
- transformers: Hugging Face library for BERT tokenizer and model.
- sklearn.preprocessing: For label encoding and one-hot encoding.
- re: Regular expressions for text preprocessing.
- pandas: For handling CSV data.
- random and numpy: For data manipulation and random sampling.
- nltk.corpus: For stopwords (requires downloading NLTK data).
- sklearn.model_selection: For train-test splitting.
- torch.utils.data: For creating data loaders.

Install these via pip install -r requirements.txt after creating the file as described in Installation.

To install cuda follow these instructions
[Download NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)