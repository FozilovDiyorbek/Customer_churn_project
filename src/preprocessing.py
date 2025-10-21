import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def clean_data(df):
    columns_to_drop = ["id", "CustomerId", "Surname"]
    df = df.drop(columns=columns_to_drop, errors='ignore')

    df.to_csv("data/processed/cleaned_customer_churn.csv", index=False)

    return df


if __name__ == "__main__":
    df = load_data("data/raw/customer_churn.csv")
    df_cleaned = clean_data(df)
    df_cleaned.head()