# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 11:02:52 2024

@author: andre
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

def generate_random_step_function_Bc(duration=1000, max_value=0.2, min_value=-0.2):
    num_steps = np.random.randint(20, 100)
    times = np.sort(np.random.uniform(0, duration, num_steps))
    values = np.random.uniform(min_value, max_value, num_steps)
    times = np.concatenate([times, [duration]])
    values = np.concatenate([values, [values[-1]]])
    return times, values, num_steps

def generate_random_step_function_Pd(duration=1000, max_value=0.4, min_value=-0.4):
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
    # Parametri
    V1 = 1
    M = 1
    Dg = 0.1
    Dl = 1
    tau = 1
    k = 0.25
    xl = 0.5  
    
    duration_Bc = 1000
    times_Bc, values_Bc, num_steps_Bc = generate_random_step_function_Bc(duration=duration_Bc)
    print(num_steps_Bc)
    duration_Pd = 1000
    times_Pd, values_Pd, num_steps_Pd = generate_random_step_function_Pd(duration=duration_Pd)
    print(num_steps_Pd)
    def system(x, t):
        return [
            1/M * (Pd(t, times_Pd, values_Pd) - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
            x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd(t, times_Pd, values_Pd)),
            (1/tau) * (-x[2]**2 * (1/xl - Bc(t, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd(t, times_Pd, values_Pd))
        ]
    
    Pd0 = np.random.uniform(0, 0.7)
    Pm = Pd0
    
    # Define the system of equations
    def equations(x):
        return [
            1/M * (Pm - (V1 * x[2] / xl) * np.sin(x[1]) - Dg * x[0]),
            x[0] - (1/Dl) * ((V1 * x[2] / xl) * np.sin(x[1]) - Pd0),
            (1/tau) * (-x[2]**2 * (1/xl - Bc(0, times_Bc, values_Bc)) + (V1 * x[2] / xl) * np.cos(x[1]) - k * Pd0)
        ]

    # Initial guess
    x0 = [0, 0, 1]

    # Solve the system of equations with fsolve
    x_0 = fsolve(equations, x0)
    
    t = np.linspace(0, 1000, 2000) 
    solution = odeint(system, x_0, t)
    
    # Generate Pd and Bc values for testing
    Pd_values_test = np.array([Pd(ti, times_Pd, values_Pd) for ti in t])
    Bc_values_test = np.array([Bc(ti, times_Bc, values_Bc) for ti in t])
    
    Results_matrix = np.column_stack((Pd_values_test, Bc_values_test, solution))  # prime due colonne gli input, ultime 3 output

    plt.plot(t, Results_matrix[:, -3:])
    plt.xlabel('Time [s]')
    plt.legend(['omega', 'delta', 'v'])
    plt.grid()
    plt.show()
    
    plt.plot(t, Results_matrix[:, 0])
    plt.xlabel('Time [s]')
    plt.legend(['Pd'])
    plt.grid()
    plt.show()
    
    plt.plot(t, Results_matrix[:, 1])
    plt.xlabel('Time [s]')
    plt.legend(['Bc'])
    plt.grid()
    plt.show()
    
    if train == 0:   
        C = Results_matrix
        print(f"Train Simulation {train}")
    else:
        C = np.vstack((C, Results_matrix))
        print(f"Train Simulation {train}")

np.save('results_matrix_100_100.npy', C)