# MoneyMap-AI

**MoneyMap-AI** is an AI-powered tool that categorizes bank transactions into categories and subcategories using a BERT-based model. It helps users organize financial data efficiently through training and prediction workflows, accessible via a command-line interface (CLI).


## 💡 Motivation
My primary motivation for creating this project was to provide an alternative to existing solutions by emphasizing two critical values:

1. Self-hosting: By allowing users to host the solution on their infrastructure, this project offers complete control over data, operations, and customization. Self-hosting eliminates the dependency on third-party services which might shut down, change their terms, or even introduce pricing that might not be favorable to all.
2. Privacy-friendly: In an era where every click, every view, and even every scroll is tracked, privacy has become a scarce commodity. This project is designed with privacy at its core. No unnecessary data collection, no sneaky trackers, and no third-party analytics. Your data stays yours, and that's how it should be.


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

## 📈 Performance
Here are my results after using main.csv dataset (62,793 lines of data) for 2 epochs. This took around 10 hours to complete based on my hardware. This is without using data augmentation to double the size due to time restaints.
![cat_modelV1](https://github.com/j-convey/BankTextCategorizer/assets/85854964/f457198d-4de0-4ef2-b7eb-3f30d6c14d58)

#### Flexibility in Categorization
However, we understand that every user might have specific needs, and the default categories might not fit everyone. You have the flexibility to modify, add, or remove categories and subcategories as per your requirements.

#### How to Customize:
Update the Dictionary: Modify the categories dictionary in the code with your desired categories and subcategories. The key should be the main category, and the values should be a list containing the subcategories.

Update Training Data: It's crucial that once you modify the categories and subcategories, you also need to change the training data. Ensure that the data has labels corresponding to your new categories and subcategories.

Re-Train the Model: With the updated categories and training data, re-run the main() function to train the model on the new data.

By following these steps, you can easily customize the categorization to suit your personal or business needs.

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


## 🔮 Future Improvements
- Web interface and mobile client
- Refining the data preprocessing pipeline.
- Fine-tuning the BERT model for better performance.


## 🤝 Contribute
1. Consider contributing to the project.
2. Add a Github Star to the project.
3. Post about the project on X.