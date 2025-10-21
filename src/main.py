from preprocessing import load_data, clean_data
from feature_engineering import create_preprocessing_pipeline
from model_training import train_model
import os

def main():
    df = load_data("data/raw/customer_churn.csv")
    df = clean_data(df)
    create_preprocessing_pipeline(df, target_column="Exited")
    best_model = train_model(df, target_column="Exited")
    print("Model training completed and best model saved.")
    return best_model

if __name__ == "__main__":
    main()