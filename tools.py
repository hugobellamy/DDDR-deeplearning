import pickle
import torch
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, X, y):
        if len(X) != len(y):
            print(
                f"Error length of feature array, {len(X)}, does not match length of value array, {len(y)}"
            )
        self.X = X
        self.y = y
        self.feat_size = len(X[0])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        X_tensor = torch.tensor(self.X[idx], dtype=torch.float32)
        y_tensor = torch.tensor(self.y[idx], dtype=torch.float)

        return X_tensor, y_tensor


def load_data(dataset_id, half=0, test=False):
    data = pickle.load(open("converted_data/" + str(dataset_id) + ".pkl", "rb"))
    sub_data = data[half]
    if test:
        dose_response = data[2]  # 3 is for training, 2 is for testing
    else:
        dose_response = data[3]
    X, y, y_uncert = sub_data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    # test data is ignored here as is uncertainty for now
    # CHANGE LATER
    return (
        Dataset(X_train, y_train),
        Dataset(X_test, y_test),
        [X_train, y_train],
        [X_test, y_test],
    )
