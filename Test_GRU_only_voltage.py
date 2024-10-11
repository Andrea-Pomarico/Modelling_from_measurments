# -*- coding: utf-8 -*-
"""
This code takes as input the GRU model and test it with a new simulation

@author: andre
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import json

#%% Generating testing data
def generate_random_step_function_Bc(duration=3000, max_value=0.2, min_value=-0.2):
    num_steps = np.random.randint(20, 100)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

def generate_random_step_function_Pd(duration=3000, max_value=0.4, min_value=-0.4):
    num_steps = np.random.randint(20, 100)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

def Bc(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]

def Pd(t, times, values):
    if t < times[0]:
        return values[0]
    for i in range(len(times) - 1):
        if times[i] <= t < times[i + 1]:
            return values[i]
    return values[-1]

for train in range(1):
    V1 = 1
    M = 1
    Dg = 0.1
    Dl = 1
    tau = 1
    k = 0.25
    xl = 0.5
    
    duration_Bc = 3000
    times_Bc, values_Bc, num_steps_Bc = generate_random_step_function_Bc(duration=duration_Bc)
    duration_Pd = 3000
    times_Pd, values_Pd, num_steps_Pd = generate_random_step_function_Pd(duration=duration_Pd)

    def system(x, t):
        return [
            1/M * (Pd(t, times_Pd, values_Pd) - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
            x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd(t, times_Pd, values_Pd)),
            (1/tau) * (-x[2]**2 * (1/xl - Bc(t, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd(t, times_Pd, values_Pd))
        ]

    Pd0 = np.random.uniform(0, 0.7)
    Pm = Pd0
    x0 = [0, 0, 1]

    def equations(x):
        return [
            1/M * (Pm - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
            x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd0),
            (1/tau) * (-x[2]**2 * (1/xl - Bc(0, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd0)
        ]

    x_0 = fsolve(equations, x0)

    t = np.linspace(0, 3000, 6000)
    solution = odeint(system, x_0, t)

    Pd_values_test = np.array([Pd(ti, times_Pd, values_Pd) for ti in t])
    Bc_values_test = np.array([Bc(ti, times_Bc, values_Bc) for ti in t])

    Results_matrix = np.column_stack((Pd_values_test, Bc_values_test, solution))

# GRU test
# Scaling
    scaler_input = MinMaxScaler()
    scaler_output = MinMaxScaler()

    inputs = Results_matrix[:, [0, 1, 4]]  # Seleziona Pd, Bc, V
    outputs = Results_matrix[:, 4].reshape(-1, 1)  # Solo V, come output

    inputs_normalized = scaler_input.fit_transform(inputs)
    outputs_normalized = scaler_output.fit_transform(outputs)

    # Definisci look_back
    look_back = 50

    def create_sequences(inputs, outputs, look_back):
        X, y = [], []
        for i in range(len(inputs) - look_back):
            X.append(inputs[i:i + look_back])
            y.append(outputs[i + look_back])
        return np.array(X), np.array(y)

    inputs_seq, outputs_seq = create_sequences(inputs_normalized, outputs_normalized, look_back)

    inputs_seq = torch.Tensor(inputs_seq)
    outputs_seq = torch.Tensor(outputs_seq).reshape(-1, 1)

    class GRUModel(nn.Module):
        def __init__(self, input_size, hidden_size, output_size, num_layers=1):
            super(GRUModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)  
            out, _ = self.gru(x, h0)
            out = self.fc(out[:, -1, :])  
            return out
# load the best model
    with open('best_hyperparameters.json', 'r') as f:
        best_params = json.load(f)

    model = GRUModel(input_size=3, hidden_size=best_params['hidden_size'], output_size=1,
                      num_layers=best_params['num_layers'])

    model.load_state_dict(torch.load('gru_model_voltage_best.pth'))
    model.eval()

    with torch.no_grad():
        predictions = model(inputs_seq)

    predictions_np = predictions.numpy()

    predictions_descaled = scaler_output.inverse_transform(predictions_np)
    true_values_aligned = scaler_output.inverse_transform(outputs_seq.numpy())
# COmpute the max error
    max_error = np.max(np.abs(predictions_descaled - true_values_aligned))
    print(f'Max Error: {max_error:.6f}')

    plt.figure(figsize=(12, 6))
    plt.plot(predictions_descaled, label='Predicted Values', alpha=0.7)
    plt.plot(true_values_aligned, label='True Values', alpha=0.7)
    plt.title('True vs Predicted Values on Results_matrix.npy')
    plt.xlabel('Time Step')
    plt.ylabel('Voltage (V)')
    plt.legend()
    plt.xlim(0, len(predictions_descaled))
    plt.grid()
    plt.show()
# Compute the MAE
    MAE = np.mean(np.abs(predictions_descaled - true_values_aligned) / np.abs(true_values_aligned)) * 100
    print(f'MAE%: {MAE:.2f}%')
