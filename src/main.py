import argparse
import torch
from model import BertModel
from train import TrainModel

from data_prep import DataPreprocessor
from load import Load_Models
from predict import Predict

def setup_device():
    """Set up the device (CPU or GPU)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device

def run_training(data_path):
    """Handle the training logic."""
    learning_rate = 1e-5
    epochs = 2
    batch_size = 64

    # Initialize data preprocessor
    data_obj = DataPreprocessor(data_path)
    num_categories = data_obj.num_categories
    num_subcategories = data_obj.num_subcategories
    print(f"Number of categories: {num_categories}")
    print(f"Number of subcategories: {num_subcategories}")

    # Prepare data for training
    cat_train_dataloader, cat_val_dataloader, sub_train_dataloader, sub_val_dataloader = data_obj.prepare_for_training()

    # Model initialization
    model = BertModel(num_categories, num_subcategories)

    # Train subcategory model (you can add category model training similarly if needed)
    sub_category_model = TrainModel(model, 'subcategory', sub_train_dataloader, sub_val_dataloader, epochs, learning_rate)
    history = sub_category_model.train()
    sub_category_model.plot_training_history(history)

    print("Training completed successfully.")

def run_prediction(model_path):
    """Handle the prediction logic."""
    csv_output_name = "fastftest.CSV"

    # Load model
    loader = Load_Models()
    loaded_category_model = loader.load_model(model_path)
    print("Model loaded successfully")

    data_obj = DataPreprocessor('data/predict/FastFood.csv')
    predict_dataloader = data_obj.prepare_DATA()

    # Run prediction
    device = setup_device()
    model = Predict(loaded_category_model, predict_dataloader, csv_output_name)
    model.run_prediction()

    print("Prediction completed successfully.")

def main():
    """Main function to parse CLI arguments and execute the appropriate command."""
    parser = argparse.ArgumentParser(description="CLI for training or predicting with a model.")
    parser.add_argument('command', choices=['train', 'predict'], help="Command to execute: 'train' or 'predict'")
    parser.add_argument('path', help="Path to data (for train) or model (for predict)")

    args = parser.parse_args()

    if args.command == 'train':
        run_training(args.path)
    elif args.command == 'predict':
        run_prediction(args.path)

if __name__ == "__main__":
    main()
