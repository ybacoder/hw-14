from joblib import load
from preprocess import prep_data
import pandas as pd

def predict_from_csv(csv_path):

    df = pd.read_csv(csv_path)
    X, y = prep_data(df)

    reg = load("reg.joblib")

    predictions = reg.predict(X)

    return predictions

if __name__ == "__main__":
    from sklearn.metrics import mean_squared_error
    predictions = predict_from_csv("fish_holdout_demo.csv")
    truth = pd.read_csv("fish_holdout_demo.csv")["Weight"].values
    mse = mean_squared_error(truth, predictions)
    print(mse)