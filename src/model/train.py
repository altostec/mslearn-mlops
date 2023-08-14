# Import libraries

import argparse
import glob
import os

import pandas as pd

# Agrego mlflow
import mlflow

from sklearn.linear_model import LogisticRegression

# Agrego train_test_split p/la func split_data
from sklearn.model_selection import train_test_split


# define functions
def main(args):
    # DONE-TO DO: enable autologging
    mlflow.autolog()

    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    train_model(args.reg_rate, X_train, X_test, y_train, y_test)


def get_csvs_df(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    csv_files = glob.glob(f"{path}/*.csv")
    if not csv_files:
        raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)


# DONE-TO DO: add function to split data
def split_data(df):
    columns = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'floors', 'price', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age', 'Diabetic']
    df = df.loc[:, columns]
    df.head(10)

    # Features and Target
    features = ['Pregnancies', 'PlasmaGlucose', 'DiastolicBloodPressure', 'TricepsThickness', 'floors', 'price', 'SerumInsulin', 'BMI', 'DiabetesPedigree', 'Age']
    X = df.loc[:, features]
    y = df.loc[:, ['Diabetic']]

    return train_test_split(X, y, random_state=0, train_size=.75)


def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data',
                        type=str)
    parser.add_argument("--reg_rate", dest='reg_rate',
                        type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)

    # parse args
    args = parse_args()

    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
