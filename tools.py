import pickle
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle


class Dataset:
    def __init__(self, X, y, y_uncert, dose_response):
        if len(X) != len(y):
            print(
                f"Error length of feature array, {len(X)}, does not match length of value array, {len(y)}"
            )

        new_X = []
        new_response = []
        new_dose = []
        y_uncert_new = []
        for i in range(len(X)):
            for dr in dose_response[i]:
                new_X.append(X[i])
                new_dose.append(dr[0])
                new_response.append(dr[1])
                if not np.isfinite(dr[0]):
                    print(dr[0])
                y_uncert_new.append(y_uncert[i])
        y_uncert_new = np.array(y_uncert_new) / np.amax(y_uncert_new)
        y_uncert_new = np.array([np.exp(-x) for x in y_uncert_new])
        print(np.amin(y_uncert_new))
        self.X, self.dose, self.response, self.y_uncert = shuffle(
            new_X, new_dose, new_response, y_uncert_new
        )

        self.feat_size = len(X[0])

    def __len__(self):
        return len(self.dose)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        dose = torch.tensor(self.dose[idx], dtype=torch.float)
        response = torch.tensor(self.response[idx], dtype=torch.float)
        y_uncert_tensor = torch.tensor(self.y_uncert[idx], dtype=torch.float)
        return X_tensor, dose, response, y_uncert_tensor


def load_data(dataset_id, half=0, test=False):
    print(dataset_id)
    with open("converted_data/"+str(dataset_id)+".pkl", "rb") as f:
        data = pickle.load(f)

    # data is
    # [basic_data, bayes_data, dose_response, use_dose_response, sigma_estimate]
    # dose_response and use_dose_response are the same
    # they varied at an earlier stage when we were making semi-real dataset
    sub_data = data[half]
    dose_response = data[2]  # 3 is for training, 2 is for testing

    X, y, y_uncert = sub_data
    (
        X_train,
        X_test,
        y_train,
        y_test,
        uncert_train,
        uncert_test,
        dose_response_train,
        dose_response_test,
    ) = train_test_split(X, y, y_uncert, dose_response, test_size=0.2, random_state=42)
    # replace uncert test because we dont want to do a weighted calculation at test time
    uncert_test = np.ones(len(y_test))
    return (
        Dataset(X_train, y_train, uncert_train, dose_response_train),
        Dataset(X_test, y_test, uncert_test, dose_response_test),
        [X_train, y_train],
        [X_test, y_test],
    )


class weighted_expwise_loss(nn.Module):
    def __init__(self, power=1):  # 0 might be a better default
        self.power_val = power
        super().__init__()

    def forward(self, ec50, dose, response, weights=None):
        if weights is None:
            weights = torch.tensor([np.ones(len(dose))])
        weights = weights**self.power_val
        pred_responses = hill_equation(ec50, dose)
        mean_differences = (pred_responses - response) ** 2
        return torch.dot(weights.flatten(), mean_differences.flatten()) / weights.sum()


def hill_equation(neg_log_EC50s, doses):
    neg_log_EC50s = torch.clamp(neg_log_EC50s, min=-10, max=10)
    EC50s = 10 ** (-neg_log_EC50s)
    return (100 * doses) / (EC50s + doses)  # Hill equation with slope = 1
