import model as r_model
import tools as tools
from torch.utils.data import DataLoader
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

torch.autograd.set_detect_anomaly(True)


hidden_dim = 300  # Number of nodes in each hidden layer
learning_rate = 8e-5
batch_size = 30
power_val = 0.33


# standard loss
train_data, test_data, train_data_alt, test_data_alt, train_dataloader_standard = (
    tools.load_data(624474)
)
feature_length = train_data.feat_size
input_dim = feature_length  # Size of your Morgan fingerprints
model = r_model.RegressionModel(input_dim, hidden_dim, 1)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
Lossfn0 = tools.weighted_expwise_loss(power_val)
Lossfn = tools.mse_loss()
r_model.train_loop(
    train_dataloader, model, learning_rate=learning_rate, loss_fn=Lossfn0
)
# performance on test set
print(
    "new version",
    r_model.test_loop(test_data, model, loss_fn=Lossfn0),
    r_model.test_loop(test_data, model, loss_fn=Lossfn),
)
# save model
torch.save(model.state_dict(), "models/1902_testing.pth")

# standard loss
feature_length = train_data.feat_size
input_dim = feature_length  # Size of your Morgan fingerprints
model = r_model.RegressionModel(input_dim, hidden_dim, 1)
train_dataloader = DataLoader(train_data, batch_size=batch_size)
r_model.train_loop(
    train_dataloader_standard, model, learning_rate=learning_rate, loss_fn=Lossfn
)
# performance on test set
print(
    "classic",
    r_model.test_loop(test_data, model, loss_fn=Lossfn0),
    r_model.test_loop(test_data, model, loss_fn=Lossfn),
)
# save model
torch.save(model.state_dict(), "models/1902_testing.pth")

rf_baseline = RandomForestRegressor(max_features=0.3)
rf_baseline.fit(train_data_alt[0], train_data_alt[1])

print(
    "Random forest baseline",
    r_model.test_loop(test_data, rf_baseline, loss_fn=Lossfn0, sklearn=True),
    r_model.test_loop(test_data, rf_baseline, loss_fn=Lossfn, sklearn=True),
)
