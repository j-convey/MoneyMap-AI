# Prerequisites
-  Python 3.9 or later


# CLI Usage
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
