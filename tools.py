import pickle
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn


class Dataset:
    def __init__(self, X, y, y_uncert):
        if len(X) != len(y):
            print(
                f"Error length of feature array, {len(X)}, does not match length of value array, {len(y)}"
            )
        self.X = X
        self.y = y
        self.feat_size = len(X[0])
        self.y_uncert = y_uncert

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float)
        y_uncert_tensor = torch.tensor(self.y_uncert[idx], dtype=torch.float)

        return X_tensor, y_tensor, y_uncert_tensor


def load_data(dataset_id, half=0, test=False):
    data = pickle.load(open("converted_data/" + str(dataset_id) + ".pkl", "rb"))
    sub_data = data[half]
    if test:
        dose_response = data[2]  # 3 is for training, 2 is for testing
    else:
        dose_response = data[3]
    X, y, y_uncert = sub_data
    X_train, X_test, y_train, y_test, uncert_train, uncert_test = train_test_split(
        X, y, y_uncert, test_size=0.2, random_state=42
    )
    # replace uncert test because we dont want to do a weighted calculation at test time
    uncert_test = np.ones(len(y_test))
    # test data is ignored here as is uncertainty for now
    # CHANGE LATER
    return (
        Dataset(X_train, y_train, uncert_train),
        Dataset(X_test, y_test, uncert_test),
        [X_train, y_train],
        [X_test, y_test],
    )


def weighted_loss(y_pred, y_true, weights=None):
    # convert weights to a pytorch tensor
    if weights is None:
        weights = [np.ones(len(y_pred))]
    mean_differences = (y_pred - y_true) ** 2
    return torch.dot(weights.flatten(), mean_differences.flatten()) / weights.sum()


class weighted_expwise_loss(nn.Module):
    def __init__(self, power=1):  # 0 might be a better default
        self.power_val = power
        super().__init__()

    def forward(self, y_pred, y_true, weights=None):
        if weights is None:
            weights = torch.tensor([np.ones(len(y_pred))])
        weights = weights**self.power_val
        mean_differences = (y_pred - y_true) ** 2
        return torch.dot(weights.flatten(), mean_differences.flatten()) / weights.sum()
