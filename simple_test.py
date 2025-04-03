import model as r_model
import tools as tools
from torch.utils.data import DataLoader
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

hidden_dim = 300  # Number of nodes in each hidden layer
learning_rate = 0.001
batch_size = 50

power_val = 0.1

train_data, test_data, train_data_alt, test_data_alt = tools.load_data(1902)
feature_length = train_data.feat_size
input_dim = feature_length  # Size of your Morgan fingerprints
model = r_model.RegressionModel(input_dim, hidden_dim, 1)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
Lossfn = tools.weighted_expwise_loss(power_val)

r_model.train_loop(train_dataloader, model, loss_fn=Lossfn)

# performance on test set
print("test set mse", r_model.test_loop(test_data, model, loss_fn=Lossfn))
# save model
torch.save(model.state_dict(), "models/1902_testing.pth")

rf_baseline = RandomForestRegressor(max_features=0.3)
rf_baseline.fit(train_data_alt[0], train_data_alt[1])

preds = rf_baseline.predict(test_data_alt[0])
print("Random forest baseline", mean_squared_error(test_data_alt[1], preds))

# sample mean baseline
y_pred = np.ones(len(test_data_alt[1])) * np.mean(train_data_alt[1])
print("Sample mean baseline", mean_squared_error(test_data_alt[1], y_pred))
