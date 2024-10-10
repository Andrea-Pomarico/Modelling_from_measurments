# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 10:37:41 2024

@author: andre
"""
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import time
from itertools import product

# Load the data
data = np.load('results_matrix_100_100.npy')  # with noise

# Prepare the data
inputs = data[:, [0, 1, 4]]  # Pd, Bc, V (3 input features)
outputs = data[:, 4]  # Just voltage (Vn+1)

look_back = 50  # Look-back period

# Normalize inputs and outputs using MinMaxScaler
input_scaler = MinMaxScaler()
output_scaler = MinMaxScaler()

inputs_normalized = input_scaler.fit_transform(inputs)
outputs_normalized = output_scaler.fit_transform(outputs.reshape(-1, 1))

def create_sequences(inputs, outputs, look_back):
    X, y = [], []
    for i in range(len(inputs) - look_back):
        X.append(inputs[i:i + look_back])
        y.append(outputs[i + look_back])
    return np.array(X), np.array(y)

# Create sequences with normalized data
inputs_seq, outputs_seq = create_sequences(inputs_normalized, outputs_normalized, look_back)

# Split the data into training (80%), validation (10%), and test (10%) sets
train_idx = int(len(inputs_seq) * 0.8)
val_idx = int(len(inputs_seq) * 0.9)

train_inputs = inputs_seq[:train_idx]
val_inputs = inputs_seq[train_idx:val_idx]
test_inputs = inputs_seq[val_idx:]

train_outputs = outputs_seq[:train_idx]
val_outputs = outputs_seq[train_idx:val_idx]
test_outputs = outputs_seq[val_idx:]

# Convert the datasets into PyTorch tensors
train_inputs = torch.Tensor(train_inputs)
train_outputs = torch.Tensor(train_outputs.reshape(-1, 1))
val_inputs = torch.Tensor(val_inputs)
val_outputs = torch.Tensor(val_outputs.reshape(-1, 1))
test_inputs = torch.Tensor(test_inputs)
test_outputs = torch.Tensor(test_outputs.reshape(-1, 1))

# Create DataLoader for training and validation data
train_dataset = TensorDataset(train_inputs, train_outputs)
train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)

val_dataset = TensorDataset(val_inputs, val_outputs)
val_loader = DataLoader(val_dataset, batch_size=50)

# Define the GRU model with dropout
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout_rate=0.2):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.dropout(out[:, -1, :])
        out = self.fc(out)
        return out

# Hyperparameter grid
hidden_sizes = [32, 64]
num_layers = [2, 3]
dropout_rates = [0.1]
learning_rates = [0.001, 0.01]

# To track the best model
best_val_loss = float('inf')
best_params = {}

# Function to format time in hours, minutes, and seconds
def format_time(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hrs}h {mins}m {secs}s"

# Total number of combinations for grid search
total_combinations = len(hidden_sizes) * len(num_layers) * len(dropout_rates) * len(learning_rates)

# Grid search
combination_counter = 1
for hidden_size, num_layer, dropout_rate, lr in product(hidden_sizes, num_layers, dropout_rates, learning_rates):
    print(f"Training with hidden_size={hidden_size}, num_layers={num_layer}, dropout_rate={dropout_rate}, learning_rate={lr} "
          f"({combination_counter}/{total_combinations})")
    
    model = GRUModel(input_size=3, hidden_size=hidden_size, output_size=1, num_layers=num_layer, dropout_rate=dropout_rate)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_epochs = 30
    train_losses = []
    val_losses = []
    
    # Monitor training time
    start_time = time.time()

    # Training loop
    for epoch in range(num_epochs):
        epoch_start = time.time()  # Start of epoch
        model.train()
        epoch_loss = 0
        for batch_inputs, batch_outputs in train_loader:
            optimizer.zero_grad()
            predictions = model(batch_inputs)
            loss = criterion(predictions, batch_outputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_inputs, batch_outputs in val_loader:
                val_predictions = model(batch_inputs)
                loss = criterion(val_predictions, batch_outputs)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Calculate elapsed time and estimated remaining time
        epoch_end = time.time()
        elapsed_time = epoch_end - start_time
        average_epoch_time = elapsed_time / (epoch + 1)
        remaining_epochs = num_epochs - (epoch + 1)
        estimated_remaining_time = remaining_epochs * average_epoch_time

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, "
              f"Estimated Remaining Time: {format_time(estimated_remaining_time)}")

    # Check if we found a new best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_params = {
            'hidden_size': hidden_size,
            'num_layers': num_layer,
            'dropout_rate': dropout_rate,
            'learning_rate': lr,
        }
    
    print(f"Best Validation Loss so far: {best_val_loss:.6f} with parameters {best_params}")
    
    combination_counter += 1

# Save the best hyperparameters
with open('best_hyperparameters.json', 'w') as f:
    json.dump(best_params, f)

# Print the best hyperparameters found
print("Best Hyperparameters:")
print(best_params)

# Evaluate the model with the best parameters on the test set
final_model = GRUModel(input_size=3, hidden_size=best_params['hidden_size'], output_size=1,
                        num_layers=best_params['num_layers'], dropout_rate=best_params['dropout_rate'])
criterion = nn.L1Loss()
optimizer = optim.Adam(final_model.parameters(), lr=best_params['learning_rate'])

# Combine training and validation datasets
full_inputs = np.concatenate((train_inputs.numpy(), val_inputs.numpy()), axis=0)
full_outputs = np.concatenate((train_outputs.numpy(), val_outputs.numpy()), axis=0)

full_dataset = TensorDataset(torch.Tensor(full_inputs), torch.Tensor(full_outputs))
full_loader = DataLoader(full_dataset, batch_size=50, shuffle=True)

# Train the final model
num_epochs = 30
for epoch in range(num_epochs):
    epoch_start = time.time()  # Start of epoch
    final_model.train()
    epoch_loss = 0
    for batch_inputs, batch_outputs in full_loader:
        optimizer.zero_grad()
        predictions = final_model(batch_inputs)
        loss = criterion(predictions, batch_outputs)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    epoch_end = time.time()
    elapsed_time = epoch_end - start_time
    average_epoch_time = elapsed_time / (epoch + 1)
    remaining_epochs = num_epochs - (epoch + 1)
    estimated_remaining_time = remaining_epochs * average_epoch_time

    print(f"Final Model - Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / len(full_loader):.6f}, "
          f"Estimated Remaining Time: {format_time(estimated_remaining_time)}")

# Save the trained model
torch.save(final_model.state_dict(), 'gru_model_voltage_best.pth')

# Evaluate the final model on the test set
final_model.eval()
test_loss = 0
with torch.no_grad():
    test_predictions = final_model(test_inputs)
    test_loss = criterion(test_predictions, test_outputs)

test_loss_value = test_loss.item()
print(f"Test Loss: {test_loss_value:.6f}")

predictions_np = test_predictions.numpy()
predictions_descaled = output_scaler.inverse_transform(predictions_np)
true_values_aligned = output_scaler.inverse_transform(test_outputs.numpy())

max_error = np.max(np.abs(predictions_descaled - true_values_aligned))
print(f'Max Error on Test Set: {max_error:.6f}')

MAE = np.mean(np.abs(predictions_descaled - true_values_aligned) / np.abs(true_values_aligned)) * 100
print(f'MAE% on Test Set: {MAE:.2f}%')

plt.figure(figsize=(12, 6))
plt.plot(predictions_descaled, label='Predicted Values', alpha=0.7)
plt.plot(true_values_aligned, label='True Values', alpha=0.7)
plt.title('True vs Predicted Values on Test Set')
plt.xlabel('Time Step')
plt.ylabel('Voltage (V)')
plt.legend()
plt.xlim(0, len(predictions_descaled))
plt.grid()
plt.show()











