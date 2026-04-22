import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str, test_size: float, random_state: int):

    df = pd.read_csv(path)
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])
    y = df["Survived"]

    return train_test_split(X, y, test_size=test_size, random_state=random_state)
