# make model, fit and predict

# Import necessary libraries from PyTorch
import torch.nn as nn  # nn contains the building blocks for neural networks
import torch.optim as optim  # optim contains optimization algorithms like Adam or SGD
import torch


class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the layers of the neural network.
        Args:
            input_size (int): The number of features in the input data (1024 for Morgan fingerprints).
            hidden_size (int): The number of nodes (neurons) in each hidden layer (300 in your case).
            output_size (int): The number of output values (1 for a standard regression problem).
        """
        super(
            RegressionModel, self
        ).__init__()  # Call the initializer of the parent class (nn.Module)

        # Define the layers sequentially
        # Layer 1: Input layer to first hidden layer
        self.layer_1 = nn.Linear(input_size, hidden_size)
        # Layer 2: First hidden layer to second hidden layer
        self.layer_2 = nn.Linear(hidden_size, hidden_size)
        # Layer 3: Second hidden layer to third hidden layer
        self.layer_3 = nn.Linear(hidden_size, hidden_size)
        # Layer 4: Third hidden layer to fourth hidden layer
        self.layer_4 = nn.Linear(hidden_size, hidden_size)
        # Layer 5: Fourth hidden layer to fifth hidden layer
        self.layer_5 = nn.Linear(hidden_size, hidden_size)
        # Output Layer: Fifth hidden layer to the final output
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Define an activation function
        # ReLU (Rectified Linear Unit) is a common choice for hidden layers.
        # It introduces non-linearity, allowing the model to learn complex patterns.
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Defines the forward pass of the network.
        This function specifies how input data flows through the layers.
        Args:
            x (torch.Tensor): The input tensor (batch of data).
        Returns:
            torch.Tensor: The output prediction(s) from the network.
        """
        # Pass input through layer 1, then apply ReLU activation
        x = self.layer_1(x)
        x = self.relu(x)

        # Pass through layer 2, then apply ReLU
        x = self.layer_2(x)
        x = self.relu(x)

        # Pass through layer 3, then apply ReLU
        x = self.layer_3(x)
        x = self.relu(x)

        # Pass through layer 4, then apply ReLU
        x = self.layer_4(x)
        x = self.relu(x)

        # Pass through layer 5, then apply ReLU
        x = self.layer_5(x)
        x = self.relu(x)

        # Pass through the output layer
        # For regression, we typically *don't* apply an activation function
        # to the final output layer, as we want the raw predicted value.
        x = self.output_layer(x)

        return x


def train_loop(dataloader, model, learning_rate=0.001, loss_fn=None):
    if loss_fn is None:
        # If no loss function is provided, use Mean Squared Error
        loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, dose, response, y_uncert, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        dose = dose.unsqueeze(1)
        response = response.unsqueeze(1)
        loss = loss_fn(pred, dose, response, y, y_uncert)
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def test_loop(dataloader, model, loss_fn=None, sklearn=False):
    test_loss = 0
    if loss_fn is None:
        # If no loss function is provided, use Mean Squared Error
        loss_fn = nn.MSELoss()
    with torch.no_grad():
        for X, dose, response, y_uncert, y in dataloader:
            # Make predictions
            if sklearn:
                pred = model.predict([X])
                pred = torch.tensor(pred).float()
            else:
                pred = model(X)  # Use the trained model

            # Reshape y to match pred's shape [batch_size, 1]
            dose = dose.view(1)
            response = response.view(1)
            y = y.view(1)
            y_uncert = y_uncert.view(1)

            # Calculate the loss for the current batch
            batch_loss = loss_fn(pred, dose, response, y, y_uncert)
            test_loss += batch_loss.item()  # Accumulate the loss

    return test_loss / len(dataloader)
