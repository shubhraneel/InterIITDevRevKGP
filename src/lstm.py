import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl
from . import base_model

# Define the input and output layers of the model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Linear(hidden_size, num_classes)

# Define the LSTM layers of the model
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        start_index = self.fc1(out)
        end_index = self.fc2(out)
        return start_index, end_index

    def __train__(self, model, dataloader):
        # Define the loss function and optimizer for the model
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters())
        print("Starting training")
        # Train the model
        num_epochs=1
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(dataloader, 0):
                #TODO: Get the inputs and labels
                inputs, labels = data

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                start_index, end_index = model(inputs)
                loss = criterion(start_index, labels[:, 0]) + criterion(end_index, labels[:, 1])

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Print the statistics
                running_loss += loss.item()
                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss = 0.0

        print('Finished Training')

    def __inference__(self, model, dataloader):
        # Test the model
        with torch.no_grad():
            correct = 0
            total = 0
            for data in dataloader:
                # Get the inputs and labels
                inputs, labels = data

                # Forward pass
                start_index, end_index = model(inputs)
                _, predicted_start = torch.max(start_index.data, 1)
                _, predicted_end = torch.max(end_index.data, 1)

                # Calculate the performance metrics
                total += labels.size(0)
                correct += (predicted_start == labels[:, 0]).sum().item()
                correct += (predicted_end == labels[:, 1]).sum().item()

        print('Accuracy of the model on the test set: %d %%' % (100 * correct / total))

