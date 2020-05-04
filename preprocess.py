import math
import pandas as pd

def prep_data(df):

    # df = df.assign(hw=df["Height"] * df["Width"]) # Ed code
    df = df\
        .assign(girth=2 * math.pi * ( ( (df["Height"] / 2) ** 2 + (df["Width"] / 2) ** 2) / 2) ** 0.5)\
        .assign(avg_length=(df["Length1"] + df["Length2"] + df["Length3"]) / 3)

    X = pd.get_dummies(df[["Species", "avg_length", "Length1", "Length2", "Length3", "Height", "Width", "girth"]], drop_first=True).values
    y = df["Weight"].values

    return X, y

if __name__ == "__main__":
    
    df = pd.read_csv("fish_participant.csv")
    print(df.head())

    print(prep_data(df))