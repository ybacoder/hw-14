import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.pipeline import make_pipeline

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import cross_validate

from sklearn.metrics import mean_squared_error

from joblib import dump
from preprocess import prep_data


df = pd.read_csv("fish_participant.csv")

X, y = prep_data(df)

## Null Model
dr = DummyRegressor()
dr_mean = cross_validate(dr, X, y, scoring="neg_mean_squared_error")["test_score"].mean()

## Linear Regression
lr = LinearRegression(fit_intercept=False)
lr_mean = cross_validate(lr, X, y, scoring="neg_mean_squared_error")["test_score"].mean()

## Decision Tree
dt = DecisionTreeRegressor()
dt_mean = cross_validate(dt, X, y, scoring="neg_mean_squared_error")["test_score"].mean()

## Random Forest
rf = RandomForestRegressor()
rf_mean = cross_validate(rf, X, y, scoring="neg_mean_squared_error")["test_score"].mean()

rf.fit(X, y)

dump(rf, "reg.joblib")

if __name__ == "__main__":
    print(dr_mean, lr_mean, dt_mean, rf_mean)