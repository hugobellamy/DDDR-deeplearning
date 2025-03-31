import pickle


def load_data(dataset_id, half=0, test=False):
    data = pickle.load(open("converted_data/" + str(dataset_id) + ".pkl", "rb"))
    sub_data = data[half]
    if test:
        dose_response = data[2]  # 3 is for training, 2 is for testing
    else:
        dose_response = data[3]
    X, y, y_uncert = sub_data
    return X, y, y_uncert, dose_response
