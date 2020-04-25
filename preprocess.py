import math

def prep_data(df):

    df = df.assign(section_area=math.pi * df["Height"]/2 * df["Width"]/2)
        #.assign(avg_length=sum(df[Length1], df[Length2], df[Length3]) / 3)

    # # ED EXAMPLE
    # X = df[["Height", "Width", "hw"]].values
    # y = df["Weight"].values

    # return X, y

    X = df[["Species", "Length1", "section_area"]].values
    y = df["Weight"].values

    return X, y

if __name__ == "__main__":
    import pandas as pd
    
    df = pd.read_csv("fish_participant.csv")
    print(df.head())

    print(prep_data(df))